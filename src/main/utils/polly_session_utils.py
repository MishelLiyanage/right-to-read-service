import boto3
import logging

logger = logging.getLogger(__name__)

def initialize_polly():
    try:
        # Try to use the specific profile first
        session = boto3.Session(profile_name='123233845129_DevOpsUser', region_name='us-east-1')
        polly_client = session.client('polly')
        logger.info("Using AWS profile: 123233845129_DevOpsUser")
        return polly_client
    except Exception as e:
        logger.warning(f"Could not use AWS profile '123233845129_DevOpsUser': {e}")
        logger.info("Falling back to default AWS credentials")
        # Fall back to default credentials (environment variables, IAM role, etc.)
        try:
            polly_client = boto3.client('polly', region_name='us-east-1')
            return polly_client
        except Exception as e2:
            logger.error(f"Could not initialize AWS Polly client: {e2}")
            # Return None - the service will handle this gracefully
            return None
