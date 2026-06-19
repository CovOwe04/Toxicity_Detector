# Multi-Platform Toxicity & Cyberbullying Detection Pipeline

## Overview

This project builds a scalable text moderation system designed to detect toxic language, cyberbullying, hate speech, and nuanced harmful content (including sarcasm and disguised profanity). The system targets real-world platforms such as forums, gaming chats, and educational communities where traditional keyword filters fail under adversarial input.

The objective is to move from static rule-based moderation to a contextual, ML-driven classification pipeline.

---

## Problem Statement

Conventional moderation systems rely heavily on keyword matching and blacklist filters. These approaches are easily bypassed using:

* Leetspeak (e.g., “h4te”)
* Obfuscation and spacing tricks
* Sarcastic or contextual abuse
* Implicit identity-based harassment

This creates unsafe digital environments, increases moderation workload, and degrades user trust and engagement.

---

## Dataset

Primary dataset:

* Jigsaw Toxic Comment Classification Challenge

Key characteristics:

* ~159K Wikipedia talk page comments
* Multi-label targets:

  * toxic
  * severe_toxic
  * obscene
  * threat
  * insult
  * identity_hate
* Noisy, real-world text with slang, typos, and adversarial formatting

Preprocessing strategy:

* Text normalization (leetspeak correction, cleanup of obfuscated tokens)
* Tokenization and sequence standardization
* Class imbalance handling (weighted loss / focal loss strategy)

---

## Model Architecture

The system uses a hybrid deep learning pipeline:

### 1. GRU-Based Fast Filter

* Lightweight recurrent model
* Handles obvious toxicity and spam at low latency
* Acts as first-pass screening layer

### 2. Transformer-Based Context Engine

* Fine-tuned Transformer encoder (context-aware classification)
* Handles ambiguous, sarcastic, and context-heavy inputs
* Produces final classification scores across all toxicity labels

### Output Layer

* Multi-label sigmoid classification
* Independent probability scores per toxicity class

---

## System Pipeline

1. User submits text input
2. Preprocessing layer normalizes and tokenizes text
3. GRU model performs rapid inference
4. Ambiguous/high-risk cases escalated to Transformer model
5. Final toxicity scores generated
6. Result logged for analytics and moderation actions

---

## Deployment Architecture

* Backend API exposed via containerized service (FastAPI-ready design)
* Scalable cloud deployment (GCP/AWS compatible)
* Event-driven inference pipeline for low latency

### Admin Dashboard

Monitoring layer built using:

* Streamlit

Dashboard features:

* Toxicity trend analytics
* Flagged content review queue
* Category-level breakdown (insults, threats, identity attacks)
* Model confidence tracking

---

## Tech Stack

* Python
* PyTorch / TensorFlow (model training)
* GRU networks
* Transformer architecture (DeBERTa/RoBERTa-style models)
* FastAPI (serving layer)
* Streamlit (analytics dashboard)
* Docker (containerization)

---

## Key Innovation

* Hybrid GRU + Transformer architecture for cost-performance balance
* Context-aware moderation instead of keyword filtering
* Multi-label classification for fine-grained toxicity detection
* Designed for real-time, high-throughput environments

---

## Goal

Deliver a production-ready moderation intelligence layer that reduces human moderation workload while improving detection accuracy for modern, evolving forms of online abuse.
