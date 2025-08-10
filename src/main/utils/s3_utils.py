import boto3
from botocore.exceptions import BotoCoreError, ClientError

def upload_file_to_s3(file_path, s3_bucket, s3_key, expires_in=3600):
    """
    Uploads a file to S3 and returns a signed URL valid for `expires_in` seconds.
    """
    s3_client = boto3.client("s3")

    try:
        # Upload the file to S3 (without public read access)
        s3_client.upload_file(
            Filename=file_path,
            Bucket=s3_bucket,
            Key=s3_key
        )

        # Generate a pre-signed URL
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': s3_bucket,
                'Key': s3_key
            },
            ExpiresIn=expires_in
        )

        return presigned_url

    except (BotoCoreError, ClientError) as error:
        print(f"Failed to upload or generate signed URL for {file_path}: {error}")
        return None
