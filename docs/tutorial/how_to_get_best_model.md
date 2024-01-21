# How to get best model?

To download the best model you can use EmbeddingStudio API:
```bash
curl -X GET http://localhost:5000/api/v1/fine-tuning/task/65844c019fa7cf0957d04758
```
where `65844c019fa7cf0957d04758` is the task ID.

If everything is OK, you will see following output:
```json
{
  "fine_tuning_method": "Default Fine Tuning Method", 
  "status": "done", 
  "best_model_url": "http://localhost:5001/get-artifact?path=model%2Fdata%2Fmodel.pth&run_uuid=571304f0c330448aa8cbce831944cfdd", 
  ...
}
```
And `best_model_url` field contains a link to the `model.pth` file.

You can download `model.pth` file by executing following command:
```bash
wget http://localhost:5001/get-artifact?path=model%2Fdata%2Fmodel.pth&run_uuid=571304f0c330448aa8cbce831944cfdd
```
