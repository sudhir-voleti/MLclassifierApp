# This script checks for required packages and installs them if they are missing.
# Run this script before launching the Shiny app to ensure all dependencies are met.

suppressPackageStartupMessages({

  # List of required packages for the application
  required_packages <- c(
    "shiny",      # Core framework for the app
    "shinyjs",    # For using javascript in shiny
    "DT",         # For interactive tables
    "caret",      # For modeling (train, dummyVars, etc.)
    "Amelia",     # For the missingness map plot
    "dplyr",      # For data manipulation
    "ggplot2",    # For plotting (used in PCA plot)
    "ROCR",       # For ROC curve calculations
    "pROC",       # For multi-class ROC curve
    "markdown",   # To render the overview.md file
    "e1071",      # A dependency for caret's SVM and Naive Bayes models
    "nnet"        # A dependency for caret's Neural Network model
  )

  # Loop through the list of packages
  for (pkg in required_packages) {
    # Check if the package is not installed
    if (!require(pkg, character.only = TRUE)) {
      # Install the package if it's not found
      install.packages(pkg, dependencies = TRUE)
      # Load the package after installation
      library(pkg, character.only = TRUE)
    }
  }

})

# Print a message to confirm completion
message("All required packages are checked and installed.")
