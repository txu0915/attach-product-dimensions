try:
    from model_mart_logging import *
except ImportError as e:
    print("In get_dimensions.py: Skipping model_mart_logging import: %s" % str(e))

try:
    from model_mart_file_utils import *
except ImportError as e:
    print("In get_dimensions.py: Skipping model_mart_file_utils import: %s" % str(e))

from utils import *
from consts import *

from pyspark.sql import SparkSession
from pyspark import StorageLevel
import pyspark.sql.functions as F
from pyspark.sql.types import *

import argparse
import json


def get_idm_attributes(day, spark):
    """
    Load the latest IDM attributes data starting with the specified day and working backwards until the data is found.

    :param day: The data to start looking for IDM data from.
    :param spark: SparkSession to read data
    :return: df with columns (oms_id, attribute_id, attribute_value)
    """
    def load_df(idm_attr_loc):
        schema = [
            StructField('oms_id', StringType()),
            StructField('attribute_id', StringType()),
            StructField('attribute_value', StringType()),
            StructField('x1', StringType()),
            StructField('x2', StringType()),
            StructField('x3', StringType()),
            StructField('x4', StringType())
        ]
        return spark.read.csv(idm_attr_loc, schema=StructType(schema), sep='|')

    idm_attributes_df, _ = load_most_recent_df(load_df, IDM_ATTRIBUTES_LOC, day, lambda d: format_day(d, '-'))

    return idm_attributes_df.select('oms_id', 'attribute_id', 'attribute_value')


def get_product_dim_attributes(product_type, config):
    """
    Given the product type, get a list of each dimension and which IDM attribute to use.
    If the specified product_type doesn't have an entry in the config, falls back to "DEFAULT"

    :param product_type: The product type of a certain product
    :param config: The config mapping from a product type to the attribute_ids to use for the width/depth/height
    :return: a list of pairs (dimension_name, dimension_IDM_attribute_id) for that particular product type
    """
    key = product_type if product_type in config else "DEFAULT"
    return [(dim_name, dim_guid) for dim_name, dim_guid in config[key].items()]


def add_dim_guids(merged_img_df, config):
    """
    For each oms_id, use its product type to figure out which attribute_ids give the width/depth/height.
    Create a row for each oms_id/dimension pair

    :param merged_img_df: The input df which has columns (oms_id, product_type)
    :param config: The config mapping from a product type to the attribute_ids to use for the width/depth/height
    :return: df with columns (oms_id, dimension_guid, dimension_name)
    """
    dim_attrs_udf = F.udf(lambda product_type: get_product_dim_attributes(product_type, config),
                          ArrayType(StructType([
                              StructField('dimension_name', StringType()),
                              StructField('dimension_guid', StringType())]
                          )))

    return merged_img_df.select('oms_id', F.explode(dim_attrs_udf('product_type')).alias('dimension')) \
        .select('oms_id', 'dimension.dimension_guid', 'dimension.dimension_name')


def get_sku_dimensions(img_with_dim_guids_df, idm_attributes_df):
    """
    Combine the dataframes mapping oms_id and dimension name to idm attribute with the idm attribute table to get
    the numerical values of the dimensions for the sku.

    :param img_with_dim_guids_df: df mapping oms_id and dimension name to idm attribute.
                                    Has columns (oms_id, dimension_guid=attribute_id, dimension_name)
    :param idm_attributes_df: df giving the IDM attribute values
    :return: df with columns (oms_id, width, height, depth)
    """
    return img_with_dim_guids_df.withColumnRenamed('dimension_guid', 'attribute_id') \
        .join(idm_attributes_df, how='left', on=['oms_id', 'attribute_id']) \
        .groupby('oms_id') \
        .pivot('dimension_name', ['width', 'height', 'depth']) \
        .agg(F.first('attribute_value')) \
        .select('oms_id', 'width', 'height', 'depth')


def get_and_save_dimensions(spark, day, config_loc, output_dir):
    # product_type, oms_id, image_guid, image_type
    logging.info("\n\nloading the merged image data")
    merged_img_df = get_merged_img_df(output_dir, day, spark)

    logging.info("\n\nloading the dimension config file")
    dimension_config_local_filepath = "dimension_config.json"
    transfer_storage_file_to_local(config_loc, dimension_config_local_filepath)
    with open(dimension_config_local_filepath, 'r') as fp:
        config = json.load(fp)

    # oms_id, dimension_guid, dimension_name
    logging.info("\n\ndetermining the dimension guids to use for each image")
    img_with_dim_guids_df = add_dim_guids(merged_img_df, config)

    # oms_id, attribute_id, attribute_value
    logging.info("\n\nloading IDM data")
    idm_attributes_df = get_idm_attributes(day, spark)

    # oms_id, width, height, depth
    logging.info("\n\ngetting the dimension values for each sku")
    sku_dimensions_df = get_sku_dimensions(img_with_dim_guids_df, idm_attributes_df)

    logging.info("\n\nsaving the dimension values")
    sku_dimensions_df.write.csv("%s/%s/%s" % (output_dir, format_day(day, '-'), DIMENSION_OUTPUT_SUFFIX))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", required=False,
                        help="The date to generate images for in format YYYY-MM-DD, defaults to the most recent day",
                        default=None, dest="date")
    parser.add_argument("--config_loc", required=False,
                        help="The config file, default\"gs://hd-datascience-np-artifacts/jim/dimension_images/config/dimension_config.json\"",
                        default="gs://hd-datascience-np-artifacts/jim/dimension_images/config/dimension_config.json",
                        dest="config_loc")
    parser.add_argument("--output_dir", required=False,
                        help="file where the output will be written, default \"gs://hd-datascience-np-data/dimension_images\"",
                        default="gs://hd-datascience-np-data/dimension_images", dest="output_dir")

    args = parser.parse_args()

    if args.date is None:
        day = get_most_recent_date(args.output_dir)
    else:
        day = parse_date(args.date, divider='-')
    logging.info("\n\nGetting dimensions for day %s" % day)

    spark = SparkSession.builder \
        .appName("Dimension Images (Get Dimensions)") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    get_and_save_dimensions(spark, day, args.config_loc, args.output_dir)

    spark.stop()
