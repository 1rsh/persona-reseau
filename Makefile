# ===========================================
# Synthetic Document → KG → Evaluation Pipeline
# ===========================================

# --- Variables ---
PYTHON = python3
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate
GENERATION_DIR = data/generated
EXTRACTED_DIR = extracted

# --- Main scripts ---
GENERATOR_SCRIPT = data.generate
EXTRACTOR_SCRIPT = main

# --- Tools ---
PIP = $(VENV)/bin/pip
PYTHON_BIN = $(VENV)/bin/python

# --- Default target ---
.DEFAULT_GOAL := run

# --- Create virtual environment ---
$(VENV)/bin/activate:
	@echo "📦 Creating virtual environment..."
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip

# --- Install dependencies ---
setup: $(VENV)/bin/activate
	@echo "⚙️ Installing dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "✅ Environment setup complete."

# --- Generate synthetic documents ---
generate:
	@echo "🧠 Generating synthetic documents..."
	@$(PYTHON_BIN) -m $(GENERATOR_SCRIPT)
	@echo "✅ Documents generated and saved under $(GENERATION_DIR)"

# --- Extract Knowledge Graphs and Evaluate ---
extract:
	@echo "🔍 Extracting Knowledge Graphs and evaluating..."
	@$(PYTHON_BIN) -m $(EXTRACTOR_SCRIPT)
	@echo "✅ Extraction and evaluation complete. Results saved to $(EXTRACTED_DIR)."

# --- Run full pipeline ---
run: generate extract
	@echo "🚀 Full pipeline completed successfully."

tidy:
	@echo "🧼 Tidying up the workspace..."
	@rm -rf $(GENERATION_DIR)/*
	@rm -rf $(EXTRACTED_DIR)/*
	@echo "✅ Workspace tidied."

# --- Clean generated and extracted outputs ---
clean:
	@echo "🧹 Cleaning up generated data..."
	@rm -rf $(VENV)
	@rm -rf $(GENERATION_DIR)/*
	@rm -rf $(EXTRACTED_DIR)/*
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleaned all outputs."

