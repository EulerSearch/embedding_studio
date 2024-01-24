# Data source

Currently, Embedding tudio only supports S3 storage. If your data resides on a different storage system, you'll need to
implement your own [DataLoader](https://github.com/EulerSearch/embedding_studio/blob/main/embedding_studio/embeddings/data/loaders/data_loader.py).

Ensure that the machine running the Embedding Studio worker has the necessary permissions to read from your S3 storage.
You'll either need to grant read permissions directly or use a separate role with read access.

In your [plugin](plugins.md), make sure to specify the following parameters:

* `role_arn`
* `aws_access_key_id`
* `aws_secret_access_key`

!!! Note 
    If you don't specify these parameters, Embedding Studio will use an **anonymous session**.

For detailed instructions on setting up permissions in S3, refer to
the [AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html).
    