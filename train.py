from sklearn.tree import DecisionTreeRegressor
import misc

def main():
    print("--- Training Decision Tree Regressor Model ---")
    
    # 1. Load data manually using instructions from assignment
    df = misc.load_data()
    
    # 2. Preprocess and split data
    X_train, X_test, y_train, y_test = misc.preprocess_and_split(df)
    
    # 3. Instantiate and train model
    model = DecisionTreeRegressor(random_state=42)
    trained_model = misc.train_model(model, X_train, y_train)
    
    # 4. Evaluate model
    mse_score = misc.evaluate_model(trained_model, X_test, y_test)
    
    print(f"Average MSE score on the test set: {mse_score:.4f}")

if __name__ == "__main__":
    main()
