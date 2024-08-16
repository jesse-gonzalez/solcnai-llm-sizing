# solcnai-llm-sizing

## Developing and Testing

```bash
$ task
task: [default] task --list
task: Available tasks for this project:
* docker-build:       Build the Docker image
* docker-push:        Push the Docker image
* docker-run:         Run the Docker image
* k8s-delete:         Delete the Kubernetes resources
* k8s-deploy:         Deploy the Kubernetes resources
* kind-create:        Create a Kubernetes Kind cluster
* kind-delete:        Delete the Kubernetes Kind cluster
```

## References

Inspired by:

- https://medium.com/@manuelescobar-dev/memory-requirements-for-llm-training-and-inference-97e4ab08091b
- https://huggingface.co/spaces/Vokturz/can-it-run-llm

docker build --platform linux/amd64 -t harbor.infrastructure.cloudnative.nvdlab.net/jesse/cnai-llm-sizer:latest .

docker run -p 8501:8501 harbor.infrastructure.cloudnative.nvdlab.net/jesse/cnai-llm-sizer:latest

docker tag harbor.infrastructure.cloudnative.nvdlab.net/jesse/k8s-streamlit:test harbor.infrastructure.cloudnative.nvdlab.net/jesse/cnai-llm-sizer:latest

docker push harbor.infrastructure.cloudnative.nvdlab.net/jesse/cnai-llm-sizer:latest

kubectl apply -f k8s/.

kubectl get po,svc,ingress