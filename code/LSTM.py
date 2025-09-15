import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.manifold import TSNE
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# === Beat Type Mapping (AAMI) ===
aami_classes = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

def load_mitdb_data(record_list, data_path):
    signals, labels = [], []
    for record in record_list:
        try:
            record_path = os.path.join(data_path, record)
            signal, _ = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            ecg = signal[:, 0]
            for i in range(len(annotation.sample) - 1):
                start = annotation.sample[i]
                end = annotation.sample[i + 1]
                label = annotation.symbol[i]
                if label in aami_classes:
                    class_label = aami_classes[label]
                    beat_segment = ecg[start:end]
                    if len(beat_segment) < 300:
                        beat_segment = np.pad(beat_segment, (0, 300 - len(beat_segment)), 'constant')
                    else:
                        beat_segment = beat_segment[:300]
                    signals.append(beat_segment)
                    labels.append(class_label)
        except Exception as e:
            print(f"⚠️ Skipping {record}: {e}")
            continue
    return np.array(signals), np.array(labels)

def attention_layer(inputs):
    attention = Dense(inputs.shape[-1], activation='softmax')(inputs)
    attended = Multiply()([inputs, attention])
    return attended

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x_attn = attention_layer(x)
    x = LSTM(32)(x_attn)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_path = "E:/BMSTU/Issa/4/Erga/Seminars/6/mit-bih-arrhythmia-database-1.0.0"
    records = [f"{i:03d}" for i in range(100, 180) if os.path.exists(os.path.join(data_path, f"{i:03d}.dat"))]

    print("📥 Loading dataset...")
    X, y = load_mitdb_data(records, data_path)
    print("📊 Data shape:", X.shape, y.shape)

    # Plot class distribution
    sns.countplot(x=pd.Series(y, name='Beat Class'))
    plt.title("Class Distribution (Original)")
    plt.grid(True)
    plt.show()

    X = (X - np.mean(X)) / np.std(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights_dict = dict(enumerate(class_weights))

    print("🧠 Building model...")
    model = build_model(X.shape[1:], y_cat.shape[1])
    model.summary()

    print("🚀 Training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights_dict
    )

    print("📈 Evaluating...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\n🔬 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

    # Accuracy and Loss Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: t-SNE visualization
    print("🎨 Running t-SNE...")
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(X_test)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for cls in np.unique(y_true_classes):
        idx = y_true_classes == cls
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_encoder.classes_[cls], alpha=0.6)
    plt.title("t-SNE of LSTM Output Features")
    plt.legend()
    plt.grid(True)
    plt.show()
