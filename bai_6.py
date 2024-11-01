import random
import math

# Khởi tạo ngẫu nhiên centroids
def initialize_centroids(X, k):
    random_indices = random.sample(range(len(X)), k)
    centroids = [X[i] for i in random_indices]
    return centroids

# Tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

# Gán mỗi điểm dữ liệu vào cụm gần nhất
def assign_clusters(X, centroids):
    clusters = [[] for _ in range(len(centroids))]
    labels = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(x)
        labels.append(cluster_index)
    return clusters, labels

# Tính lại vị trí centroid của mỗi cụm
def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:  # Kiểm tra nếu cụm không rỗng
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            centroids.append(centroid)
    return centroids

# Thuật toán K-means chính
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters, labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:  # hội tụ
            break
        centroids = new_centroids
    return labels, centroids

# Tính F1-score
def f1_score_manual(true_labels, predicted_labels, k):
    contingency_matrix = [[0] * k for _ in range(k)]
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i]][predicted_labels[i]] += 1

    f1_scores = []
    for i in range(k):
        tp = contingency_matrix[i][i]
        fp = sum(contingency_matrix[j][i] for j in range(k)) - tp
        fn = sum(contingency_matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

# Tính Rand Index
def rand_index(true_labels, predicted_labels):
    tp_fp = 0
    tp_fn = 0
    tp = 0
    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):
            same_cluster_true = true_labels[i] == true_labels[j]
            same_cluster_pred = predicted_labels[i] == predicted_labels[j]
            if same_cluster_true and same_cluster_pred:
                tp += 1
            if same_cluster_true:
                tp_fn += 1
            if same_cluster_pred:
                tp_fp += 1
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = len(true_labels) * (len(true_labels) - 1) // 2 - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

# Tính entropy
def entropy(labels):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    entropy_value = 0
    for count in label_counts.values():
        p = count / len(labels)
        entropy_value -= p * math.log2(p) if p > 0 else 0
    return entropy_value

# Tính Mutual Information
def mutual_information(true_labels, predicted_labels):
    mutual_info = 0
    unique_true = set(true_labels)
    unique_pred = set(predicted_labels)
    for t in unique_true:
        for p in unique_pred:
            intersection = sum(1 for i in range(len(true_labels)) if true_labels[i] == t and predicted_labels[i] == p)
            if intersection > 0:
                p_t = sum(1 for i in range(len(true_labels)) if true_labels[i] == t) / len(true_labels)
                p_p = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == p) / len(predicted_labels)
                p_tp = intersection / len(true_labels)
                mutual_info += p_tp * math.log2(p_tp / (p_t * p_p)) if (p_t * p_p) > 0 else 0
    return mutual_info

# Tính Normalized Mutual Information (NMI)
def normalized_mutual_information(true_labels, predicted_labels):
    h_true = entropy(true_labels)
    h_pred = entropy(predicted_labels)
    mi = mutual_information(true_labels, predicted_labels)
    return mi / ((h_true + h_pred) / 2) if (h_true + h_pred) > 0 else 0

# Tính Davies-Bouldin Index
def cluster_diameter(cluster):
    n = len(cluster)
    if n <= 1:
        return 0
    distances = [euclidean_distance(cluster[i], cluster[j]) for i in range(n) for j in range(i + 1, n)]
    return sum(distances) / len(distances)

def davies_bouldin_index(X, clusters, centroids):
    db_index = 0
    for i in range(len(centroids)):
        max_ratio = 0
        for j in range(len(centroids)):
            if i != j:
                d_i = cluster_diameter(clusters[i])
                d_j = cluster_diameter(clusters[j])
                centroid_dist = euclidean_distance(centroids[i], centroids[j])
                ratio = (d_i + d_j) / centroid_dist if centroid_dist > 0 else 0
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / len(centroids) if centroids else 0

# Hàm đọc dữ liệu từ file CSV
def load_iris_data(filename):
    X = []
    true_labels = []
    label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            if len(row) < 5:  # Ensure there are at least 5 columns
                continue  # Skip this row if it's invalid
            try:
                features = list(map(float, row[:4]))  # Convert first 4 columns to float
                label = label_mapping[row[4]]  # Map the last column to label
                X.append(features)
                true_labels.append(label)
            except ValueError:
                print(f"Warning: Could not convert row to float: {row}")  # Log the invalid row
                continue  # Skip this row if conversion fails
    return X, true_labels


# Kiểm tra dữ liệu trong iris.csv
X, true_labels = load_iris_data('iris.csv')
print("First 5 rows of X:", X[:5])
print("First 5 true labels:", true_labels[:5])

# Chạy thuật toán và đánh giá
def main():
    k = 3  # Số cụm
    predicted_labels, centroids = kmeans(X, k)
    clusters, _ = assign_clusters(X, centroids)

    f1 = f1_score_manual(true_labels, predicted_labels, k)
    ri = rand_index(true_labels, predicted_labels)
    nmi = normalized_mutual_information(true_labels, predicted_labels)
    db = davies_bouldin_index(X, clusters, centroids)

    print("F1 Score:", f1)
    print("Rand Index:", ri)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Davies-Bouldin Index:", db)

if __name__ == "__main__":
    main()
