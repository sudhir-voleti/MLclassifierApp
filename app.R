#------------------------------------------------------------------#
# 1. Load all necessary packages
#------------------------------------------------------------------#
options(warn = -1)
suppressPackageStartupMessages({
  library(shiny)
  library(shinyjs)
  library(DT)
  library(caret)
  library(Amelia)
  library(dplyr)
  library(ROCR)
  library(pROC)
  library(ggplot2)
})


#------------------------------------------------------------------#
# 2. Helper Functions for Plotting and Modeling
#------------------------------------------------------------------#

# --- PCA Plot Function ---
pca_plot <- function(y, X) {
  if (is.numeric(y)) {
    y <- as.character(paste0('y_', y))
  }
  X_num <- X %>% dplyr::select(where(is.numeric))
  
  if (ncol(X_num) < 2) {
    # Return a message plot if not enough numeric columns for PCA
    p <- ggplot() +
      annotate("text", x = 0, y = 0, label = "PCA requires at least two numeric features.", size = 5) +
      theme_void()
    return(plot(p))
  }
  
  a1 <- princomp(X_num, cor = TRUE)$scores[, 1:2]
  a2 <- data.frame(y = y, x1 = a1[, 1], x2 = a1[, 2])
  
  p <- ggplot(data = a2, aes(x = x1, y = x2, colour = factor(y))) +
    geom_point(size = 4, shape = 19, alpha = 0.6) +
    xlab("PCA Component 1") + ylab("PCA Component 2") +
    theme_bw() +
    labs(colour = "Target")
  
  plot(p)
}

# --- ROC Curve for Binary Classification ---
plot_roc_binary <- function(model0, df0_test) {
  # Ensure target variable 'y' is a factor
  df0_test$y <- as.factor(df0_test$y)
  
  # Check if model has probability prediction support
  if (!"prob" %in% model0$modelInfo$type) {
     message("This model does not support probability predictions for ROC.")
     return(NULL)
  }

  model.probs <- predict(model0, df0_test[, -1], type = "prob")
  
  pred <- ROCR::prediction(model.probs[, 2], df0_test$y)
  perf <- ROCR::performance(pred, "tpr", "fpr")
  auc_ROCR <- ROCR::performance(pred, measure = "auc")
  auc <- auc_ROCR@y.values[[1]]
  
  plot(perf, colorize = TRUE, main = "ROC Curve")
  legend("bottomright",
         bty = "n",
         legend = paste0("AUC Score: ", round(auc, 3)),
         cex = 1)
  
  return(auc)
}

# --- ROC Curve for Multiclass Classification ---
plot_roc_multi <- function(model0, df0_test) {
  model_preds <- predict(model0, df0_test[, -1], type = "prob")
  y_test <- as.factor(df0_test$y)
  
  # Ensure column names of predictions match factor levels
  colnames(model_preds) <- make.names(colnames(model_preds))
  
  a0 <- multiclass.roc(y_test ~ model_preds)
  auc_val <- auc(a0)
  
  rs <- a0[['rocs']]
  plot.roc(rs[[1]], main = "Multiclass ROC")
  if(length(rs) > 1){
    sapply(2:length(rs), function(i) lines.roc(rs[[i]], col = i))
  }
  
  legend("bottomright",
         bty = "n",
         legend = paste0("Mean AUC Score: ", round(auc_val, 3)),
         cex = 1)
  
  return(auc_val)
}


# --- Generic ROC Plotting Function ---
plot_roc_gen <- function(model0, df0_test) {
  # Ensure 'y' column is present
  if (!"y" %in% colnames(df0_test)) {
      message("Test data must contain a 'y' column.")
      return(NULL)
  }
  
  num_levels <- length(unique(df0_test$y))
  
  if (num_levels > 2) {
    plot_roc_multi(model0, df0_test)
  } else {
    plot_roc_binary(model0, df0_test)
  }
}


# --- Core Model Training Function ---
run_model_training <- function(df0,
                               kfoldcv_ui = 5,
                               train_propn_ui = 0.7,
                               model_selected_ui = "lg_reg",
                               svm_type = NULL) {
  
  set.seed(123)
  if (!is.factor(df0$y)) {
    df0$y <- as.factor(df0$y)
  }
  
  inTrain <- createDataPartition(y = df0$y, p = train_propn_ui, list = FALSE)
  df0_train <- df0[inTrain, ]
  df0_test <- df0[-inTrain, ]
  
  train_control <- trainControl(
    method = "repeatedcv",
    number = kfoldcv_ui,
    repeats = 3,
    classProbs = TRUE,
    savePredictions = TRUE,
    summaryFunction = if (length(levels(df0$y)) > 2) multiClassSummary else twoClassSummary
  )
  
  metric <- if (length(levels(df0$y)) > 2) "Mean_AUC" else "ROC"
  
  model_fit <- NULL
  
  tryCatch({
    if (model_selected_ui == "lg_reg") {
      model_fit <- train(
        x = df0_train[, -1], y = df0_train[, 1],
        method = 'glm', family = 'binomial',
        trControl = train_control, metric = metric
      )
    } else if (model_selected_ui == "svm") {
      method_name <- switch(svm_type,
                            "SVM_linear_fixed" = "svmLinear",
                            "SVM_linear_grid" = "svmLinear",
                            "SVM_RBF" = "svmRadial",
                            "SVM_polynomial" = "svmPoly")
      
      tuneGrid <- NULL
      if (svm_type == "SVM_linear_grid") {
        tuneGrid <- expand.grid(C = seq(0.1, 2, length = 10))
      }
      
      model_fit <- train(
        y ~ ., data = df0_train, method = method_name,
        trControl = train_control, preProcess = c("center", "scale"),
        tuneGrid = tuneGrid, metric = metric
      )
    } else if (model_selected_ui == "nb") {
      model_fit <- train(
        y ~ ., data = df0_train, method = 'nb',
        trControl = train_control, metric = metric
      )
    } else if (model_selected_ui == "nn") {
      model_fit <- train(
        y ~ ., data = df0_train, method = 'nnet',
        trControl = train_control, metric = metric,
        tuneGrid = expand.grid(size = c(5, 10), decay = c(0.1, 0.01)),
        preProcess = c("center", "scale"),
        trace = FALSE
      )
    }
  }, error = function(e) {
    message(paste("Model training failed:", e$message))
    return(NULL)
  })
  
  if (!is.null(model_fit)) {
    return(list(model_fit, df0_train, df0_test))
  } else {
    return(NULL)
  }
}

#------------------------------------------------------------------#
# 3. UI Definition (REVISED)
#------------------------------------------------------------------#
ui <- fluidPage(
  useShinyjs(), # Initialize shinyjs
  titlePanel(div(img(src = "logo.png", align = 'right', height = '50px'), "ML Algos for Classification")),
  
  sidebarPanel(
    h4("Core Controls"),
    # This panel will always be visible
    fileInput("tr_data", "Upload Training Data (CSV)", accept = ".csv"),
    fileInput("test_data", "Upload Prediction Data (CSV, optional)", accept = ".csv"),
    
    # Conditional panel for data setup and processing
    uiOutput("data_setup_ui"),
    
    # Conditional panel for model configuration, appears after data is processed
    uiOutput("model_config_ui")
  ),
  
  mainPanel(
    tabsetPanel(
      id = "main_tabs",
      tabPanel("Overview",
               includeMarkdown("overview.md") # Ensure you have an overview.md file
      ),
      # --- NEW TAB ADDED HERE ---
      tabPanel("1. View Raw Data",
               h4("Raw Training Data Preview (Top 10 Rows)"),
               p("Review your uploaded data below. Based on this, go to the sidebar on the left to select your categorical (non-metric) variables under 'Step 1: Preprocessing'."),
               hr(),
               DT::dataTableOutput("raw_data_preview")
      ),
      # --- SUBSEQUENT TABS RENUMBERED ---
      tabPanel("2. Data Setup & Preprocessing",
               h4("Instructions"),
               p("1. Upload your training data."),
               p("2. In the sidebar, select variables you identify as categorical for one-hot encoding."),
               p("3. Click 'Process Data' to create dummy variables."),
               p("4. In the sidebar, select the final target (Y) and predictor (X) variables for your model."),
               p("5. (Optional) Upload a prediction dataset. It will be processed using the same rules."),
               hr(),
               h4("Processed Data Preview (After One-Hot Encoding)"),
               DT::dataTableOutput("processed_data_preview")
      ),
      tabPanel("3. Data Exploration",
               h4("Structure of Final Model Data"),
               verbatimTextOutput("data_str"),
               hr(),
               h4("Missingness Map (Raw Training Data)"),
               plotOutput("miss_plot"),
               hr(),
               h4("PCA Plot (Final Model Data)"),
               plotOutput("pca_plot")
      ),
      tabPanel("4. Model Training & Results",
               h4("Distribution of Target Variable (Y)"),
               verbatimTextOutput("tar_dis"),
               hr(),
               h4("Model Performance (from Cross-Validation)"),
               verbatimTextOutput("mod_res"),
               hr(),
               h4("Final Model Summary"),
               helpText("Note: Training may take a few moments."),
               verbatimTextOutput("mod_sum")
      ),
      tabPanel("5. Performance Plots",
               h4("ROC-AUC Curve (on Test Set)"),
               plotOutput("roc"),
               hr(),
               h4("Confusion Matrix (on Test Set)"),
               verbatimTextOutput('conf_test'),
               hr(),
               h4("Confusion Matrix (from Cross-Validation)"),
               verbatimTextOutput('conf_train')
      ),
      tabPanel("6. Prediction",
               helpText("Note: Predictions are run on the uploaded prediction data after applying the same transformations as the training data."),
               DT::dataTableOutput("test_op"),
               downloadButton("download_pred", "Download Predictions")
      )
    )
  )
)

#------------------------------------------------------------------#
# 4. Server Logic
#------------------------------------------------------------------#
server <- function(input, output, session) {
  
  # Reactive values to store data states
  data_store <- reactiveValues(
    raw_train = NULL,
    raw_test = NULL,
    processed_train = NULL,
    ohe_recipe = NULL,
    final_data = NULL,
    model_results = NULL
  )
  
  # --- Data Loading ---
  observeEvent(input$tr_data, {
    data_store$raw_train <- read.csv(input$tr_data$datapath, stringsAsFactors = FALSE)
  })
  
  observeEvent(input$test_data, {
    data_store$raw_test <- read.csv(input$test_data$datapath, stringsAsFactors = FALSE)
  })
  
    # --- Dynamic UI for Data Setup (REVISED LOGIC) ---
  output$data_setup_ui <- renderUI({
    req(data_store$raw_train)
    
    # Get column names from raw data
    cols <- names(data_store$raw_train)
    
    tagList(
      hr(),
      h4("Step 1: Define Variables"),
      # UI for selecting the target variable FIRST
      selectInput('sel_y', "Select Y (Target Variable)", choices = cols, multiple = FALSE),
      
      # This UI will be rendered by another function below
      uiOutput("categorical_selector_ui"),
      
      hr(),
      actionButton("process_data_btn", "Process Data", class = "btn-primary"),
      hr(),
      # This UI appears after processing
      uiOutput("predictor_selection_ui")
    )
  })

  # --- New UI renderer for selecting categorical PREDICTORS ---
  output$categorical_selector_ui <- renderUI({
    req(data_store$raw_train, input$sel_y)
    
    # Choices for categorical variables are all columns EXCEPT the selected Y
    available_choices <- setdiff(names(data_store$raw_train), input$sel_y)
    
    selectInput("categorical_vars", "Select Categorical Predictors for OHE",
                choices = available_choices,
                multiple = TRUE)
  })
  
    # --- One-Hot Encoding and Data Processing (REVISED LOGIC) ---
  observeEvent(input$process_data_btn, {
    req(data_store$raw_train, input$sel_y)
    
    withProgress(message = 'Processing data...', value = 0.2, {
      
      # 1. Isolate Y variable and predictor (X) data from the raw training set
      y_col_data <- data_store$raw_train[, input$sel_y, drop = FALSE]
      x_data_train <- data_store$raw_train[, setdiff(names(data_store$raw_train), input$sel_y), drop = FALSE]
      
      incProgress(0.2, detail = "Creating OHE recipe...")
      
      # 2. Create OHE recipe ONLY from categorical predictors, if any are selected
      if (!is.null(input$categorical_vars) && length(input$categorical_vars) > 0) {
        formula_str <- paste("~", paste(input$categorical_vars, collapse = " + "))
        ohe_recipe <- dummyVars(as.formula(formula_str), data = x_data_train, fullRank = TRUE)
        data_store$ohe_recipe <- ohe_recipe
        
        # 3. Apply recipe to predictor data
        dummied_train <- as.data.frame(predict(ohe_recipe, newdata = x_data_train))
        
        # 4. Identify original numeric predictors
        numeric_vars <- setdiff(names(x_data_train), input$categorical_vars)
        
        # 5. Combine Y, original numeric X's, and dummied X's
        data_store$processed_train <- cbind(y_col_data, x_data_train[, numeric_vars, drop = FALSE], dummied_train)
        
      } else {
        # If no categorical variables were selected, just combine Y and X
        data_store$processed_train <- cbind(y_col_data, x_data_train)
        data_store$ohe_recipe <- NULL # No recipe
      }
      
      incProgress(0.3, detail = "Applying to prediction data...")
      
      # 6. Process test data if it exists
      if (!is.null(data_store$raw_test) && !is.null(data_store$ohe_recipe)) {
        df_test <- data_store$raw_test
        # This will now work, as Y is not in the recipe or the test data
        dummied_test <- as.data.frame(predict(data_store$ohe_recipe, newdata = df_test))
        
        numeric_vars_test <- setdiff(names(df_test), input$categorical_vars)
        data_store$processed_test <- cbind(df_test[, numeric_vars_test, drop = FALSE], dummied_test)
      } else if (!is.null(data_store$raw_test)) {
        data_store$processed_test <- data_store$raw_test
      }
      
      shinyjs::alert("Data processing complete! Please select your final predictor variables below.")
    })
  })
  
   # --- Dynamic UI for Final Predictor (X) Selection ---
  output$predictor_selection_ui <- renderUI({
    req(data_store$processed_train)
    
    # Choices for predictors are all columns in the processed data EXCEPT the target
    predictor_choices <- setdiff(names(data_store$processed_train), input$sel_y)
    
    tagList(
      h4("Step 2: Final Predictor Selection"),
      selectInput("sel_x", "Select Final X Variables for Model",
                  choices = predictor_choices,
                  multiple = TRUE,
                  selected = predictor_choices)
    )
  })
  
  # --- Dynamic UI for Model Configuration ---
  output$model_config_ui <- renderUI({
    # Show this only after variables are selected
    req(input$sel_y, input$sel_x)
    
    tagList(
      hr(),
      h4("Step 3: Model Configuration"),
      radioButtons("model_sel", "Select Model",
                   choices = c("Logistic Regression" = "lg_reg",
                               "Naive Bayes" = "nb",
                               "SVM" = "svm",
                               "Neural Networks" = "nn")),
      uiOutput("svm_type_ui"),
      sliderInput("tr_per", "Percentage of training data for split", min = 0.5, max = 0.9, value = 0.7, step = 0.05),
      sliderInput("kfold", "Number of CV folds", min = 3, max = 10, value = 5, step = 1),
      actionButton("train_model_btn", "Train Model", class = "btn-success")
    )
  })
  
  output$svm_type_ui <- renderUI({
    req(input$model_sel == "svm")
    selectInput('svm_type', label = "Select SVM Type",
                choices = c("Linear SVM" = "SVM_linear_fixed",
                            "Linear SVM (Grid Search)" = "SVM_linear_grid",
                            "Nonlinear SVM (RBF)" = "SVM_RBF",
                            "Nonlinear SVM (Polynomial)" = "SVM_polynomial"))
  })
  
  # --- Final Data Assembly for Model ---
  observeEvent(input$train_model_btn, {
    req(data_store$processed_train, input$sel_y, input$sel_x)
    
    df <- data_store$processed_train
    
    # Check if target is in selected features
    if (input$sel_y %in% input$sel_x) {
        shinyjs::alert("Error: Target variable cannot also be a predictor variable.")
        return()
    }

    # Reorder columns to have 'y' first, then predictors
    final_cols <- c(input$sel_y, input$sel_x)
    
    # Ensure all selected columns exist
    if (!all(final_cols %in% names(df))) {
        shinyjs::alert("Error: One or more selected columns not found in processed data.")
        return()
    }
    
    final_df <- df[, final_cols]
    names(final_df)[1] <- "y"
    
    # Ensure all predictor columns are numeric
    non_numeric_predictors <- names(which(sapply(final_df[, -1, drop = FALSE], Negate(is.numeric))))
    if(length(non_numeric_predictors) > 0){
        shinyjs::alert(paste("Error: All predictor columns must be numeric. The following are not:", paste(non_numeric_predictors, collapse=", ")))
        return()
    }
    
    data_store$final_data <- final_df
  })
  
  # --- Model Training Event ---
  model_results_reactive <- eventReactive(input$train_model_btn, {
    req(data_store$final_data)
    
    withProgress(message = 'Training model in progress...', {
      run_model_training(
        df0 = data_store$final_data,
        kfoldcv_ui = input$kfold,
        train_propn_ui = input$tr_per,
        model_selected_ui = input$model_sel,
        svm_type = input$svm_type
      )
    })
  })
  
    # --- NEW --- Tab for Raw Data View ---
  output$raw_data_preview <- DT::renderDataTable({
    req(data_store$raw_train)
    DT::datatable(
      head(data_store$raw_train, 10),
      caption = "Displaying the first 10 rows of the uploaded training data.",
      options = list(scrollX = TRUE, pageLength = 10, searching = FALSE, dom = 't') # dom = 't' shows only the table itself
    )
  })
  
  # --- Tab 1 Outputs: Processed Data Preview ---
  output$processed_data_preview <- DT::renderDataTable({
    req(data_store$processed_train)
    DT::datatable(head(data_store$processed_train, 100), options = list(scrollX = TRUE, pageLength = 5))
  })
  
  # --- Tab 2 Outputs: Data Exploration ---
  output$data_str <- renderPrint({
    req(data_store$final_data)
    str(data_store$final_data)
  })
  
  output$miss_plot <- renderPlot({
    req(data_store$raw_train)
    Amelia::missmap(data_store$raw_train)
  })
  
  output$pca_plot <- renderPlot({
    req(data_store$final_data)
    y <- data_store$final_data$y
    X <- data_store$final_data[, -1]
    pca_plot(y, X)
  })
  
  # --- Tab 3 Outputs: Model Training & Results ---
  output$tar_dis <- renderPrint({
    req(data_store$final_data)
    round(prop.table(table(data_store$final_data$y)), 3)
  })
  
  output$mod_res <- renderPrint({
    req(model_results_reactive())
    model_results_reactive()[[1]]$results
  })
  
  output$mod_sum <- renderPrint({
    req(model_results_reactive())
    model_results_reactive()[[1]]
  })
  
  # --- Tab 4 Outputs: Performance Plots ---
  output$roc <- renderPlot({
    req(model_results_reactive())
    model <- model_results_reactive()[[1]]
    test_data <- model_results_reactive()[[3]]
    plot_roc_gen(model, test_data)
  })
  
  output$conf_test <- renderPrint({
    req(model_results_reactive())
    model <- model_results_reactive()[[1]]
    test_data <- model_results_reactive()[[3]]
    
    preds <- predict(model, test_data[, -1])
    confusionMatrix(preds, test_data$y)
  })
  
  output$conf_train <- renderPrint({
    req(model_results_reactive())
    # This shows the resampled (cross-validation) confusion matrix
    confusionMatrix(model_results_reactive()[[1]])
  })
  
  # --- Tab 5 Outputs: Prediction ---
  prediction_df <- reactive({
    req(model_results_reactive(), data_store$processed_test, input$sel_x)
    
    model <- model_results_reactive()[[1]]
    processed_test_data <- data_store$processed_test
    
    # Ensure predictor columns match
    model_predictors <- model$finalModel$xNames
    
    # Check if all model predictors are in the processed test data
    missing_cols <- setdiff(model_predictors, names(processed_test_data))
    if (length(missing_cols) > 0) {
        # Create missing columns with 0
        for(col in missing_cols){
            processed_test_data[[col]] <- 0
        }
    }
    
    # Ensure order is the same
    pred_data_for_model <- processed_test_data[, model_predictors]
    
    predictions <- predict(model, newdata = pred_data_for_model)
    
    # Combine predictions with the raw test data for context
    data.frame(Prediction = predictions, data_store$raw_test)
  })
  
  output$test_op <- DT::renderDataTable({
    req(prediction_df())
    DT::datatable(prediction_df(), options = list(scrollX = TRUE, pageLength = 10))
  })
  
  output$download_pred <- downloadHandler(
    filename = function() {
      "predictions.csv"
    },
    content = function(file) {
      write.csv(prediction_df(), file, row.names = FALSE)
    }
  )
}

# Run the application
shinyApp(ui = ui, server = server)
