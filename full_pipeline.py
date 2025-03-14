import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, expr, to_date, avg, min as pyspark_min, max as pyspark_max
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
import logging


# Configurar PySpark para que use "python" en lugar de "python3"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

JDBC_URL = f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
JDBC_PROPERTIES = {
    "user": DB_CONFIG["user"],
    "password": DB_CONFIG["password"],
    "driver": "org.postgresql.Driver"
}

# Esquema definido para mejorar rendimiento
SCHEMA = StructType([
    StructField("timestamp", StringType(), False),
    StructField("price", FloatType(), False),
    StructField("user_id", IntegerType(), False)
])

def create_spark_session():
    """Crea una sesión de Spark con soporte para PostgreSQL."""
    return SparkSession.builder \
        .appName("Microbatching Pipeline") \
        .config("spark.driver.extraClassPath", "jars/postgresql-42.7.5.jar") \
        .config("spark.jars", "jars/postgresql-42.7.5.jar") \
        .config("spark.sql.shuffle.partitions", 8) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

def read_csv(file_path, spark):
    """Carga un archivo CSV y convierte el campo de fecha correctamente."""
    try:
        logger.info(f"Leyendo archivo: {file_path}")
        
        df = spark.read.csv(file_path, header=True, schema=SCHEMA)
        
        # Convertimos 'timestamp' a 'transaction_date' con el formato correcto
        df = df.withColumn("transaction_date", to_date(col("timestamp"), "M/d/yyyy")) \
               .drop("timestamp")  # Eliminamos la columna original

        return df
    except Exception as e:
        logger.error(f"Error al leer {file_path}: {str(e)}", exc_info=True)
        return None

def validate_data(df):
    """Valida los datos y reporta casuísticas de errores."""

    # Definir condiciones de error
    df = df.withColumn("error_type", expr("""
    CASE 
        WHEN price IS NULL THEN 'price_null'
        WHEN price <= 0 THEN 'price_negative_or_zero'
        WHEN transaction_date IS NULL THEN 'date_null'
        WHEN user_id IS NULL THEN 'user_id_null'
        ELSE NULL 
    END
"""))

    # Filtrar registros válidos e inválidos
    df_valid = df.filter(col("error_type").isNull()).drop("error_type")
    df_invalid = df.filter(col("error_type").isNotNull())

    # Contar tipos de errores
    error_counts = df_invalid.groupBy("error_type").agg(count("*").alias("count")).collect()

    if error_counts:
        error_message = "Se detectaron registros inválidos con las siguientes casuísticas:\n"
        for row in error_counts:
            error_message += f"- {row['error_type']}: {row['count']} registros.\n"
        logger.warning(error_message.strip())

    return df_valid, df_invalid

def compute_batch_statistics(df):
    """Calcula estadísticas del batch actual."""
    stats = df.select(
        count("*").alias("batch_count"),
        avg("price").alias("batch_avg"),
        pyspark_min("price").alias("batch_min"),
        pyspark_max("price").alias("batch_max")
    ).collect()[0]

    return stats.batch_count, stats.batch_avg, stats.batch_min, stats.batch_max

def get_previous_statistics(spark):
    """Obtiene las estadísticas acumuladas desde PostgreSQL."""
    try:
        query = "(SELECT * FROM transaction_statistics ORDER BY total_count DESC LIMIT 1) AS stats"
        stats_df = spark.read.jdbc(url=JDBC_URL, table=query, properties=JDBC_PROPERTIES)

        if stats_df.count() == 0:
            logger.warning("La tabla transaction_statistics está vacía. Inicializando valores predeterminados.")
            return 0, 0.0, None, None  # Inicializa valores si no hay datos

        stats = stats_df.collect()[0]  # Extraer la primera fila

        return (
            int(stats["total_count"]),
            float(stats["avg_price"]),
            float(stats["min_price"]) if stats["min_price"] is not None else None,
            float(stats["max_price"]) if stats["max_price"] is not None else None
        )

    except Exception as e:
        logger.error(f"Error al leer estadísticas previas: {str(e)}", exc_info=True)
        return 0, 0.0, None, None  # Retorna valores predeterminados en caso de error


import builtins  # Importamos la versión nativa de min y max

def update_statistics_in_postgres(batch_count, batch_avg, batch_min, batch_max, spark):
    """Actualiza las estadísticas acumuladas en PostgreSQL."""

    prev_count, prev_avg, prev_min, prev_max = get_previous_statistics(spark)

    # Calcular el nuevo promedio ponderado
    if prev_count == 0:
        new_avg = batch_avg  # Primer batch
    else:
        new_avg = ((prev_avg * prev_count) + (batch_avg * batch_count)) / (prev_count + batch_count)

    # Determinar nuevos min y max evitando errores con None
    min_values = [x for x in [prev_min, batch_min] if x is not None]
    new_min = builtins.min(min_values) if min_values else None

    max_values = [x for x in [prev_max, batch_max] if x is not None]
    new_max = builtins.max(max_values) if max_values else None

    new_count = prev_count + batch_count

    logger.info(f"Actualizando estadísticas -> Count: {new_count}, Avg: {new_avg:.2f}, Min: {new_min}, Max: {new_max}")

    try:
        # Crear un DataFrame con los nuevos valores
        stats_df = spark.createDataFrame([
            (new_count, new_avg, new_min, new_max)
        ], ["total_count", "avg_price", "min_price", "max_price"])

        # Sobreescribir la tabla con los nuevos valores
        stats_df.write.jdbc(url=JDBC_URL, table="transaction_statistics", mode="overwrite", properties=JDBC_PROPERTIES)
        logger.info("Estadísticas actualizadas correctamente en PostgreSQL.")

    except Exception as e:
        logger.error(f"Error al actualizar estadísticas en PostgreSQL: {str(e)}", exc_info=True)


def save_to_postgres(df, table_name):
    """Guarda los datos en PostgreSQL usando JDBC."""

    # Verificar cantidad de registros antes de escribir
    row_count = df.count()
    logger.info(f"Se encontraron {row_count} registros para insertar en {table_name}")

    try:
        df.write.jdbc(url=JDBC_URL, table=table_name, mode="append", properties=JDBC_PROPERTIES)
        logger.info(f"Datos insertados en la tabla {table_name}")
    except Exception as e:
        logger.error(f"Error al guardar en PostgreSQL: {str(e)}", exc_info=True)

def process_csv(file_path, spark):
    """Pipeline de microbatch para un archivo CSV con estadísticas."""
    df = read_csv(file_path, spark)
    if df is None or df.limit(1).count() == 0:
        logger.warning(f"Archivo {file_path} vacío o con errores, omitiendo.")
        return

    df_valid, df_invalid = validate_data(df)

    # Guardar datos válidos en PostgreSQL
    save_to_postgres(df_valid, "transactions")

    # Calcular estadísticas del batch actual
    batch_count, batch_avg, batch_min, batch_max = compute_batch_statistics(df_valid)

    # Actualizar estadísticas en PostgreSQL
    update_statistics_in_postgres(batch_count, batch_avg, batch_min, batch_max, spark)

def main():
    """Ejecuta el pipeline de microbatching."""
    spark = create_spark_session()
    file_paths = ["data/2012-1.csv", "data/2012-2.csv", "data/2012-3.csv", "data/2012-4.csv", "data/2012-5.csv"]

    for file in file_paths:
        process_csv(file, spark)

    logger.info("Pipeline completado.")
    spark.stop()

if __name__ == "__main__":
    main()