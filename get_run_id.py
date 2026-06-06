import mlflow

client = mlflow.tracking.MlflowClient()
# Mengambil run terakhir dari experiment default (ID "0")
runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)

if runs:
    print(runs[0].info.run_id)
else:
    print("Run tidak ditemukan")