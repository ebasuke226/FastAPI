FROM jupyter/scipy-notebook:latest

WORKDIR /jupyter

RUN pip install mlflow==2.8.1 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.3

COPY train_model.ipynb .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"] 