FROM python:3.8

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 8080

CMD python app.py

gcloud builds submit --tag gcr.io/cifar-clf/cifar-clf --project=cifar-clf
gcloud run deploy --image gcr.io/cifar-clf/cifar-clf --platform managed --project=cifar-clf --allow-unauthenticated
