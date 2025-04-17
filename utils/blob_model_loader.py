import os
from azure.storage.blob import BlobServiceClient

def download_model_if_needed(file_name, local_path):
    connection_str = os.getenv("AZURE_BLOB_CONNECTION")
    container = "models"

    if not os.path.exists(local_path):
        print(f"📥 Downloading {file_name} from Azure Blob Storage...")
        blob_service = BlobServiceClient.from_connection_string(connection_str)
        blob_client = blob_service.get_container_client(container).get_blob_client(file_name)

        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        print(f"✅ Model saved to: {local_path}")
    else:
        print(f"✅ Model already exists: {local_path}")
