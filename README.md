## Lightweight tag storage for sentense embeddings
  Using fastapi+chromadb+onnx(tinybert)

  To run
```
make setup
make build
make spawn_model
make up
```

  To stop
```
make clean
```
### Swagger

  localhost:8001/docs


##### special for @sergey_koryagin


#### :TODO
  design is very human and easy to use

##### do not rely on (it has not tested yet)
1.  batching is working correctly
2.  metadata 
3.  etc =)

##### bert-like models
russian: https://huggingface.co/cointegrated/rubert-tiny2
