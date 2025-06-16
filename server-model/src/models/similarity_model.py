import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

class SimilarityModel:
    def __init__(self, input_dim: int):
        self.model = self._build_model(input_dim)
    
    def _build_model(self, input_dim: int) -> Model:
        """Build a neural network model for similarity learning"""
        input_layer = Input(shape=(input_dim,))
        
        # Shared layers for both users
        x = Dense(256, activation='relu')(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        embedding = Dense(64, activation='linear', name='embedding')(x)
        
        return Model(inputs=input_layer, outputs=embedding)
    
    def compile_model(self):
        """Compile the model"""
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X_train, y_train=None, epochs=10, batch_size=32):
        """
        Train the model. Since this is content-based, we might not need explicit training,
        but the method is provided for flexibility.
        """
        # For content-based, we can use the same data as input and target
        if y_train is None:
            y_train = X_train
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def generate_embeddings(self, data):
        """Generate embeddings for the input data"""
        return self.model.predict(data)
    
    def compute_similarity_matrix(self, embeddings):
        """Compute cosine similarity matrix from embeddings"""
        normalized_embeddings = tf.math.l2_normalize(embeddings, axis=1)
        return tf.linalg.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)