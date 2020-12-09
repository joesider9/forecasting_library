import pika, uuid, time, json, joblib, os
import numpy as np

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

sys_path = '/models/'
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer) or isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.str) or isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, bool):
            return bool(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            print(obj)
            raise TypeError('Object is not JSON serializable')

class rabbit_client_data(object):
    def __init__(self ):
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='data_manager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response


class rabbit_client_nwp(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='nwp_manager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_Fuzzy_Data(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='FuzzyDatamanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_FeatSel(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='FeatSelmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response


class rabbit_client_CNN(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='CNNmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_LSTM(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='LSTMmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_MLP(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='MLPmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_RBFNN(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='RBFNNmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_RBF_CNN(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='RBF_CNN_manager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response


class rabbit_client_RBFOLS(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='RBFOLSmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_SKlearn(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='SKlearnmanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_ClustComb(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='ClusterCombinemanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

class rabbit_client_ModelComb(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='ModelCombinemanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response


class rabbit_client_Proba(object):
    def __init__(self ):
        credentials = pika.PlainCredentials('admin', 'admin')
        parameters = pika.ConnectionParameters(RABBIT_MQ_HOST,
                                               RABBIT_MQ_PORT)
        start_time = time.time()

        while True:
            # wait for rabbitmq
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except:
                print('Cannot connect yet, sleeping 5 seconds.')
                time.sleep(5)
            if time.time() - start_time > 60:
                print('Could not connect after 30 seconds.')
                exit(1)


        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)


    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, static_data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='Probamanager',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(static_data, cls=NumpyEncoder))
        while self.response is None:
            self.connection.process_data_events()
        return self.response