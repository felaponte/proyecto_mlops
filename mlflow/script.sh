#!/bin/sh

mlflow server \
--backend-store-uri mysql+pymysql://user:password@db-metadata-mlflow:3306/db \
--default-artifact-root s3://mlflows3/artifacts \
--host 0.0.0.0 \
--serve-artifacts
