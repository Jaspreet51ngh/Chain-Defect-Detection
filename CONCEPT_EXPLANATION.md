# How It Works: The Brain Behind the AI

This document explains the technical concepts of your Gold Chain Defect Detection system in simple terms. Use this to explain the project to others.

## 1. The Core Concept: One-Class Anomaly Detection
Traditional AI needs two piles of images: "Good" and "Bad" to learn the difference.
**Problem:** In the real world, defects are rare. We might have 1,000 good chains but only 5 broken ones. A normal AI cannot learn from just 5 examples.

**Solution:** We use **One-Class Learning**.
*   We **only** show the AI "Good" images.
*   The AI learns "This is what a normal chain looks like."
*   Anything that deviates from this learned "normality" is automatically flagged as an anomaly.

---

## 2. The Model: PatchCore
We are using an industry-standard algorithm called **PatchCore**. Here is how it works step-by-step:

### Step 1: The "Eye" (ResNet50 Backbone)
We use a deep learning model called **ResNet50**.
*   It is pre-trained on millions of images (ImageNet).
*   It already knows how to detect edges, curves, metal textures, and shiny patterns.
*   We do not "retrain" this part; we just use it to **extract features** from your images.

### Step 2: The "Memory Bank" (Training Phase)
When you run `main.py train`:
1.  The AI looks at your "Good" images.
2.  It doesn't look at the whole image at once. It breaks it down into thousands of tiny squares called **patches**.
3.  It converts each patch into a mathematical fingerprint (an "embedding").
4.  It saves these fingerprints into a **Memory Bank**.
    *   *Analogy:* Imagine a person memorizing every single valid link shape, reflection, and angle from the good chains.

### Step 3: The "Comparison" (Prediction Phase)
When you run `main.py predict` on a new image:
1.  The AI breaks the *new* image into patches.
2.  For **every single patch** in the new image, it asks the Memory Bank:
    > *"Have you seen a patch that looks exactly like this before?"*
3.  It measures the **distance** to the nearest matching patch in its memory.

## 3. How "OK" vs "NOT OK" is decided

### The Anomaly Score
*   **Low Distance**: "Yes, I have a patch in memory that looks 99% like this." -> **Score ≈ 0** (Normal)
*   **High Distance**: "No, I have never seen a patch with this weird jagged edge or dark gap before." -> **Score > 0** (Anomaly)

The **Anomaly Score** for the image is the score of the *most abnormal* patch found in that image.
*   If 99% of the chain is perfect, but ONE link is broken, that one broken link will have a high score.
*   Therefore, the whole image gets a high score.

### The Threshold
Since no two good chains are identical (lighting changes, slight rotation), even good chains have a small "distance" (e.g., score 1.3 or 7.1).
*   We set a **Threshold** (e.g., 7.85).
*   **Score < Threshold**: "This variance is within the normal range." -> **OK**
*   **Score > Threshold**: "This looks too different to be normal." -> **NOT OK**

---

## 4. How the Heatmap is Generated
The heatmap is just a visual map of those "distances".
*   **Blue areas**: The model found a close match in memory (Normal).
*   **Red/Orange areas**: The model could NOT find a match in memory (Anomaly).
*   Because we check every patch, we can paint exactly *where* the anomaly is.

---

## Summary for Presentation
1.  **Method**: Unsupervised Anomaly Detection (PatchCore).
2.  **Why**: Works with very little data, requires no defect samples.
3.  **How**: It builds a "memory bank" of normal textures.
4.  **Decision**: If a part of the new image is mathematically "too far" from anything in the memory bank, it is flagged as a defect.
