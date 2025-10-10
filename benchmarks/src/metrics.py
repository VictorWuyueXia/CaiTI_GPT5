# benchmarks/src/metrics.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score, precision_recall_fscore_support,
	classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

def compute_binary(y_true, y_pred):
	"""
	Compute binary classification metrics: accuracy, precision, recall, f1 for positive class (label 1).
	Returns a dictionary with metrics and a detailed classification report.
	"""
	acc = accuracy_score(y_true, y_pred)  # Overall accuracy
	prec, rec, f1, _ = precision_recall_fscore_support(
		y_true, y_pred, average="binary", pos_label=1, zero_division=0
	)  # Precision, recall, f1 for positive class
	report = classification_report(
		y_true, y_pred, digits=4, zero_division=0, output_dict=True
	)  # Full classification report as dict
	return {
		"accuracy": float(acc),
		"precision_pos1": float(prec),
		"recall_pos1": float(rec),
		"f1_pos1": float(f1),
		"report": report
	}

def compute_multiclass(y_true, y_pred):
	"""
	Compute multiclass classification metrics: accuracy, macro-averaged precision, recall, f1.
	Returns a dictionary with metrics and a detailed classification report.
	"""
	acc = accuracy_score(y_true, y_pred)  # Overall accuracy
	p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)  # Macro-averaged precision, recall, f1
	report = classification_report(
		y_true, y_pred, digits=4, zero_division=0, output_dict=True
	)  # Full classification report as dict
	return {
		"accuracy": float(acc),
		"precision_macro": float(p_macro),
		"recall_macro": float(r_macro),
		"f1_macro": float(f_macro),
		"report": report
	}

def plot_confusion(y_true, y_pred, labels, title, out_path):
	"""
	Plot and save a confusion matrix for the given true and predicted labels.
	Args:
		y_true: list/array of true labels
		y_pred: list/array of predicted labels
		labels: list of label values (for axis ticks)
		title: plot title
		out_path: file path to save the plot
	Returns:
		Confusion matrix as a nested list.
	"""
	cm = confusion_matrix(y_true, y_pred, labels=labels)  # Compute confusion matrix
	fig, ax = plt.subplots(figsize=(5,4), dpi=160)
	im = ax.imshow(cm, cmap="Blues")  # Show confusion matrix as image
	ax.set_title(title)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels([str(x) for x in labels], rotation=45)
	ax.set_yticklabels([str(x) for x in labels])
	# Annotate each cell with the count
	for (i, j), v in np.ndenumerate(cm):
		ax.text(j, i, str(v), ha="center", va="center", color="black")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	fig.tight_layout()
	# Ensure output directory exists
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)
	return cm.tolist()