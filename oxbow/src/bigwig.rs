use arrow::array::{
    ArrayRef, Float32Array, Float32Builder, StringDictionaryBuilder, UInt32Array, UInt32Builder,
};
use arrow::array::{Float64Array, Float64Builder, StringArray, UInt64Array, UInt64Builder};
use arrow::datatypes::Int32Type;
use arrow::{error::ArrowError, record_batch::RecordBatch};
use bigtools::utils::reopen::ReopenableFile;
use bigtools::{BBIRead, BigWigRead, Summary};
use noodles::core::Region;
use std::collections::HashSet;
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::batch_builder::{finish_batch, BatchBuilder};

/// A BigWig reader.
pub struct BigWigReader<R> {
    read: BigWigRead<R>,
    region: Option<(String, u32, u32)>,
    zoom_level: Option<u32>,
    zoom_summary_columns: Option<HashSet<String>>,
}

pub struct BigWigRecord<'a, Value> {
    pub chrom: &'a str,
    pub start: u32,
    pub end: u32,
    pub value: Value,
}

impl BigWigReader<ReopenableFile> {
    /// Creates a BigWig reader from a given file path.
    pub fn new_from_path(path: &str) -> std::io::Result<Self> {
        let read = BigWigRead::open_file(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(Self {
            read,
            region: None,
            zoom_level: None,
            zoom_summary_columns: None,
        })
    }
}

impl<R: Read + Seek> BigWigReader<R> {
    /// Creates a BigWig reader from a given file path.
    pub fn new(read: R) -> std::io::Result<Self> {
        let read = BigWigRead::open(read)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(Self {
            read,
            region: None,
            zoom_level: None,
            zoom_summary_columns: None,
        })
    }

    /// Specifies what region to return values overlapping for. If not called,
    /// all values will be returned.
    pub fn with_region(&mut self, region: &str) -> Result<(), ArrowError> {
        let region: Region = region
            .parse()
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        let chrom_name = region.name().to_owned();
        let region = match (region.interval().start(), region.interval().end()) {
            (Some(start), Some(end)) => {
                let start = start.get() as u32 - 1; // 1-based to 0-based
                let end = end.get() as u32;
                (chrom_name, start, end)
            }
            (Some(start), None) => {
                let start = start.get() as u32 - 1; // 1-based to 0-based
                let end = self
                    .read
                    .get_chroms()
                    .iter()
                    .find(|c| c.name == chrom_name)
                    .map(|c| c.length);
                let end = end.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("Invalid chromosome".to_string())
                })?;
                (chrom_name, start, end)
            }
            (None, Some(end)) => {
                let start = 0;
                let end = end.get() as u32;
                (chrom_name, start, end)
            }
            (None, None) => {
                let start = 0;
                let end = self
                    .read
                    .get_chroms()
                    .iter()
                    .find(|c| c.name == chrom_name)
                    .map(|c| c.length);
                let end = end.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("Invalid chromosome".to_string())
                })?;
                (chrom_name, start, end)
            }
        };
        self.region = Some(region);

        Ok(())
    }

    pub fn using_zoom(&mut self, zoom_level: u32) {
        self.zoom_level = Some(zoom_level);
    }

    /// Specifies what values to query from zoom summary data. Only useful if
    /// `using_zoom` is called.
    ///
    /// Valid columns are:
    ///   - total_items
    ///   - bases_covered
    ///   - min
    ///   - max
    ///   - sum
    ///   - sum_squares
    pub fn with_zoom_summary_columns(&mut self, columns: HashSet<String>) {
        self.zoom_summary_columns = Some(columns);
    }

    /// Returns the records in the given region as Apache Arrow IPC.
    ///
    /// If the region is `None`, all records are returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxbow::bigwig::BigWigReader;
    ///
    /// let mut reader = BigWigReader::new_from_path("sample.bigWig").unwrap();
    /// reader.with_region("sq0:1-1000").unwrap();
    /// let ipc = reader.records_to_ipc().unwrap();
    /// ```
    pub fn records_to_ipc(self) -> Result<Vec<u8>, ArrowError> {
        match self.zoom_level {
            Some(zoom_level) => self.internal_zoom_records_to_ipc(zoom_level),
            None => self.internal_records_to_ipc(),
        }
    }

    fn internal_records_to_ipc(mut self) -> Result<Vec<u8>, ArrowError> {
        let capacity = 1024;
        let mut batch_builder =
            BigWigBatchBuilder::new(capacity, Float32Array::builder(capacity), &mut self.read)?;

        macro_rules! push_value {
            ($chrom_name:expr, $value:expr) => {
                let v = $value.map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let record = BigWigRecord {
                    chrom: $chrom_name,
                    start: v.start,
                    end: v.end,
                    value: v.value,
                };
                batch_builder.push(record);
            };
        }
        match self.region {
            Some((chrom_name, start, end)) => {
                let values = self
                    .read
                    .get_interval(&chrom_name, start, end)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                for value in values {
                    push_value!(&chrom_name, value);
                }
            }
            None => {
                let chroms = self.read.get_chroms().into_iter();
                for chrom in chroms {
                    let start = 0;
                    let end = chrom.length;
                    let values = self.read.get_interval(&chrom.name, start, end).unwrap();
                    for value in values {
                        push_value!(&chrom.name, value);
                    }
                }
            }
        }
        finish_batch(batch_builder)
    }

    fn internal_zoom_records_to_ipc(mut self, zoom_level: u32) -> Result<Vec<u8>, ArrowError> {
        let capacity = 1024;
        let builder = (
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("total_items"))
                .then(|| UInt64Array::builder(capacity)),
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("bases_covered"))
                .then(|| UInt64Array::builder(capacity)),
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("min"))
                .then(|| Float64Array::builder(capacity)),
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("max"))
                .then(|| Float64Array::builder(capacity)),
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("sum"))
                .then(|| Float64Array::builder(capacity)),
            self.zoom_summary_columns
                .as_ref()
                .map_or(true, |c| c.contains("sum_Squares"))
                .then(|| Float64Array::builder(capacity)),
        );
        let mut batch_builder = BigWigBatchBuilder::new(capacity, builder, &mut self.read)?;

        macro_rules! push_value {
            ($chrom_name:expr, $value:expr) => {
                let v = $value.map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let record = BigWigRecord {
                    chrom: $chrom_name,
                    start: v.start,
                    end: v.end,
                    value: v.summary,
                };
                batch_builder.push(record);
            };
        }
        match self.region {
            Some((chrom_name, start, end)) => {
                let values = self
                    .read
                    .get_zoom_interval(&chrom_name, start, end, zoom_level)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                for value in values {
                    push_value!(&chrom_name, value);
                }
            }
            None => {
                let chroms = self.read.get_chroms().into_iter();
                for chrom in chroms {
                    let start = 0;
                    let end = chrom.length;
                    let values = self
                        .read
                        .get_zoom_interval(&chrom.name, start, end, zoom_level)
                        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                    let chrom_name = &chrom.name;
                    for value in values {
                        push_value!(chrom_name, value);
                    }
                }
            }
        }
        finish_batch(batch_builder)
    }
}

trait ValueToIpc {
    type Builder;
    type Schema: IntoIterator<Item = (&'static str, ArrayRef)>;

    fn append_value_to(self, builder: &mut Self::Builder);
    fn finish(builder: Self::Builder) -> Self::Schema;
}

impl ValueToIpc for f32 {
    type Builder = Float32Builder;
    type Schema = [(&'static str, ArrayRef); 1];

    fn append_value_to(self, builder: &mut Self::Builder) {
        builder.append_value(self)
    }

    fn finish(mut builder: Self::Builder) -> [(&'static str, ArrayRef); 1] {
        [("value", Arc::new(builder.finish()) as ArrayRef)]
    }
}

impl ValueToIpc for Summary {
    type Builder = (
        Option<UInt64Builder>,  // total items
        Option<UInt64Builder>,  // bases covered
        Option<Float64Builder>, // min
        Option<Float64Builder>, // max
        Option<Float64Builder>, // sum
        Option<Float64Builder>, // sum squares
    );
    type Schema = std::iter::Flatten<std::array::IntoIter<Option<(&'static str, ArrayRef)>, 6>>;

    fn append_value_to(self, builder: &mut Self::Builder) {
        builder.0.as_mut().map(|b| b.append_value(self.total_items));
        builder
            .1
            .as_mut()
            .map(|b| b.append_value(self.bases_covered));
        builder.2.as_mut().map(|b| b.append_value(self.min_val));
        builder.3.as_mut().map(|b| b.append_value(self.max_val));
        builder.4.as_mut().map(|b| b.append_value(self.sum));
        builder.5.as_mut().map(|b| b.append_value(self.sum_squares));
    }
    fn finish(mut builder: Self::Builder) -> Self::Schema {
        [
            builder
                .0
                .as_mut()
                .map(|b| ("total_items", Arc::new(b.finish()) as ArrayRef)),
            builder
                .1
                .as_mut()
                .map(|b| ("bases_covered", Arc::new(b.finish()) as ArrayRef)),
            builder
                .2
                .as_mut()
                .map(|b| ("min", Arc::new(b.finish()) as ArrayRef)),
            builder
                .3
                .as_mut()
                .map(|b| ("max", Arc::new(b.finish()) as ArrayRef)),
            builder
                .4
                .as_mut()
                .map(|b| ("sum", Arc::new(b.finish()) as ArrayRef)),
            builder
                .5
                .as_mut()
                .map(|b| ("sum_squares", Arc::new(b.finish()) as ArrayRef)),
        ]
        .into_iter()
        .flatten()
    }
}

struct BigWigBatchBuilder<V: ValueToIpc> {
    chrom: StringDictionaryBuilder<Int32Type>,
    start: UInt32Builder,
    end: UInt32Builder,
    value_builder: V::Builder,
}

impl<V: ValueToIpc> BigWigBatchBuilder<V> {
    pub fn new<R: Read + Seek>(
        capacity: usize,
        value_builder: V::Builder,
        read: &mut BigWigRead<R>,
    ) -> Result<Self, ArrowError> {
        let chroms: Vec<String> = read.get_chroms().iter().map(|c| c.name.clone()).collect();
        let chroms: StringArray = StringArray::from(chroms);
        Ok(Self {
            chrom: StringDictionaryBuilder::<Int32Type>::new_with_dictionary(capacity, &chroms)?,
            start: UInt32Array::builder(capacity),
            end: UInt32Array::builder(capacity),
            value_builder,
        })
    }
}

impl<V: ValueToIpc> BatchBuilder for BigWigBatchBuilder<V> {
    type Record<'a> = BigWigRecord<'a, V>;

    fn push(&mut self, record: Self::Record<'_>) {
        self.chrom.append_value(record.chrom);
        self.start.append_value(record.start);
        self.end.append_value(record.end);
        record.value.append_value_to(&mut self.value_builder)
    }

    fn finish(mut self) -> Result<RecordBatch, ArrowError> {
        RecordBatch::try_from_iter(
            [
                ("chrom", Arc::new(self.chrom.finish()) as ArrayRef),
                ("start", Arc::new(self.start.finish()) as ArrayRef),
                ("end", Arc::new(self.end.finish()) as ArrayRef),
            ]
            .into_iter()
            .chain(V::finish(self.value_builder)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::ipc::reader::FileReader;
    use arrow::record_batch::RecordBatch;

    fn read_record_batch(region: Option<&str>) -> RecordBatch {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("../fixtures/valid.bigWig");
        let mut reader = BigWigReader::new_from_path(dir.to_str().unwrap()).unwrap();
        if let Some(region) = region {
            reader.with_region(region).unwrap();
        }
        let ipc = reader.records_to_ipc().unwrap();
        let cursor = std::io::Cursor::new(ipc);
        let mut arrow_reader = FileReader::try_new(cursor, None).unwrap();
        // make sure we have one batch
        assert_eq!(arrow_reader.num_batches(), 1);
        arrow_reader.next().unwrap().unwrap()
    }

    fn read_record_batch_zoom(region: Option<&str>) -> RecordBatch {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("../fixtures/valid.bigWig");
        let mut reader = BigWigReader::new_from_path(dir.to_str().unwrap()).unwrap();
        if let Some(region) = region {
            reader.with_region(region).unwrap();
        }
        reader.using_zoom(10240);
        let ipc = reader.records_to_ipc().unwrap();
        let cursor = std::io::Cursor::new(ipc);
        let mut arrow_reader = FileReader::try_new(cursor, None).unwrap();
        // make sure we have one batch
        assert_eq!(arrow_reader.num_batches(), 1);
        arrow_reader.next().unwrap().unwrap()
    }

    #[test]
    fn test_read_all() {
        let record_batch = read_record_batch(None);
        assert_eq!(record_batch.num_rows(), 100000);
        let record_batch = read_record_batch_zoom(None);
        assert_eq!(record_batch.num_rows(), 16);
    }

    #[test]
    fn test_region_full() {
        let record_batch = read_record_batch(Some("chr17"));
        assert_eq!(record_batch.num_rows(), 100000);
        let record_batch = read_record_batch_zoom(Some("chr17"));
        assert_eq!(record_batch.num_rows(), 16);
    }

    #[test]
    fn rest_region_partial() {
        let record_batch = read_record_batch(Some("chr17:59000-60000"));
        assert_eq!(record_batch.num_rows(), 4);
        let record_batch = read_record_batch_zoom(Some("chr17:59000-60000"));
        assert_eq!(record_batch.num_rows(), 1);
    }
}
