from transformers import LayoutLMv2ForSequenceClassification, LayoutLMv2Processor
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class DocumentClassifier:
    def __init__(self, model_path=None, num_labels=4, device=None):
        """
        Initializes the DocumentClassifier with a LayoutLMv2 model and processor.

        Args:
            model_path (str): Path to a pre-trained model (if provided).
            num_labels (int): Number of classification labels.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

        if model_path:
            self.model = LayoutLMv2ForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = LayoutLMv2ForSequenceClassification.from_pretrained(
                "microsoft/layoutlmv2-base-uncased",
                num_labels=num_labels
            )

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.label_map = {  # Default label mapping
            0: "bank_account_application",
            1: "identity_document",
            2: "financial_document",
            3: "receipt"
        }

    def set_label_map(self, label_map):
        """
        Sets a custom label map for classification.
        """
        self.label_map = label_map

    def prepare_input(self, image, text, layout):
        """
        Prepares the input for the LayoutLMv2 model.

        Args:
            image (PIL.Image): Input document image.
            text (str): Extracted text from the document.
            layout (dict): Layout information (coordinates of text boxes).

        Returns:
            dict: Encoded input for the model.
        """
        encoding = self.processor(
            image,
            text,
            boxes=self._get_boxes(layout),
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return encoding.to(self.device)

    def _get_boxes(self, layout):
        """
        Converts layout information into bounding boxes.
        """
        return list(zip(layout["left"], layout["top"],
                        layout["width"], layout["height"]))

    def predict(self, image, text, layout):
        """
        Predicts the document type given the input image, text, and layout.

        Args:
            image (PIL.Image): Input document image.
            text (str): Extracted text from the document.
            layout (dict): Layout information (coordinates of text boxes).

        Returns:
            dict: Predicted document type and confidence score.
        """
        try:
            inputs = self.prepare_input(image, text, layout)
            outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()

            return {
                "document_type": self.label_map.get(pred_label, "Unknown"),
                "confidence": confidence
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"document_type": "Error", "confidence": 0.0}

    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=5e-5):
        """
        Trains the LayoutLMv2 model on the provided datasets.

        Args:
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and validation.
            learning_rate (float): Learning rate for optimizer.
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0

            for batch in tqdm(train_loader, desc="Training"):
                images, texts, layouts, labels = batch

                # Prepare inputs
                inputs = self.processor(
                    images, texts, boxes=[self._get_boxes(layout) for layout in layouts],
                    padding="max_length", max_length=512, truncation=True, return_tensors="pt"
                ).to(self.device)

                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Training Loss: {avg_loss:.4f}")

            # Validation step
            self._validate(val_loader, criterion)

    def _validate(self, val_loader, criterion):
        """
        Validates the model on the validation dataset.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, texts, layouts, labels = batch

                # Prepare inputs
                inputs = self.processor(
                    images, texts, boxes=[self._get_boxes(layout) for layout in layouts],
                    padding="max_length", max_length=512, truncation=True, return_tensors="pt"
                ).to(self.device)

                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)

                total_loss += loss.item()

                # Accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")