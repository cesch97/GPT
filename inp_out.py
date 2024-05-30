import boto3
import os
from botocore.exceptions import ClientError
import hashlib


def read_file(file_path, mode='r', bucket=None):
    if bucket is not None:
        s3 = boto3.client('s3')
        try:
            # Assuming "bucket" is in the format "s3://bucket-name/prefix"
            parts = bucket.replace('s3://', '').split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            key = f"{prefix}/{file_path}".strip('/')  # Avoid leading slash
            # Get the MD5 hash of the file
            md5 = s3.head_object(Bucket=bucket_name, Key=key)['ETag']
            # Check if local file exists and has the same MD5 hash
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    local_md5 = hashlib.md5(f.read()).hexdigest()
            else:
                local_md5 = None
            if local_md5 != md5.strip('"'):
                obj = s3.get_object(Bucket=bucket_name, Key=key)
            else:
                print(f"Local file {file_path} is up to date")
                with open(file_path, mode) as f:
                    return f.read()
        except ClientError as e:
            # Handle the NoSuchKey error or other S3 errors
            print(f"Error accessing {file_path}: {e}")
            return None
        # Write data to file, create directories if necessary
        os.makedirs(os.path.dirname(file_path), exist_ok=True)        
        if 'b' in mode:
            with open(file_path, 'wb') as f:
                f.write(obj['Body'].read())
        else:
            with open(file_path, 'w') as f:
                f.write(obj['Body'].read().decode('utf-8'))
    # Read from local file
    if os.path.exists(file_path):
        with open(file_path, mode) as f:
            return f.read()
    else:
        return None


def write_file(file_path, data, mode='w', bucket=None):
    if bucket is not None:
        s3 = boto3.client('s3')
        parts = bucket.replace('s3://', '').split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        key = f"{prefix}/{file_path}".strip('/')
        s3.put_object(Body=data if 'b' in mode else data.encode('utf-8'), Bucket=bucket_name, Key=key)
    # Write to local file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        f.write(data)



def append_to_file(file_path, data, mode='a', bucket=None):
    if 'b' in mode and not isinstance(data, bytes):
        raise ValueError("Binary mode requires data to be bytes")
    elif 'b' not in mode and not isinstance(data, str):
        raise ValueError("Text mode requires data to be str")
    
    if bucket is not None:
        # For S3 bucket, read existing content and append new data
        existing_data = read_file(file_path, 'rb' if 'b' in mode else 'r', bucket)
        if existing_data is None:
            existing_data = b'' if 'b' in mode else ''
        new_data = existing_data + ('\n' + data if 'b' not in mode else b'\n' + data)
        write_file(file_path, new_data, 'wb' if 'b' in mode else 'w', bucket)
    else:
        # Direct append for local files
        write_file(file_path, '\n' + data if 'b' not in mode else b'\n' + data, mode, bucket)


def log_to_file(message, logger=None, bucket=None, log_file="log.log"):
    if (bucket is None) and (logger is not None):
        logger.info(message)
    else:
        # Ensure message has a newline when directly appending to log file
        append_to_file(log_file, message, 'a', bucket)
