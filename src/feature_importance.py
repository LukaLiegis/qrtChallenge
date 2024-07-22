import matplotlib.pyplot as plt
import pandas as pd

def analyze_feature_importances(model, X, top_n=20):
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

    # Select top N features
    top_features = feature_importances.head(top_n)

    plt.figure(figsize=(12, 8))
    plt.bar(range(top_n), top_features['importance'])
    plt.xticks(range(top_n), top_features['feature'], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

    return feature_importances