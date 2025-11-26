USE AgroSmart;

-- Users
INSERT INTO users (username, email, password_hash, full_name, role)
VALUES 
('farmer1', 'farmer1@example.com', 'hashed_password_1', 'John Doe', 'farmer'),
('admin1', 'admin1@example.com', 'hashed_password_2', 'Alice Smith', 'admin');

-- Crops
INSERT INTO crops (crop_name)
VALUES 
('Wheat'),
('Rice');

-- Crop Yield
INSERT INTO crop_yield (user_id, crop_name, area, production, yield)
VALUES
(1, 'Wheat', 5.5, 20.0, 3.64),
(2, 'Rice', 3.0, 15.0, 5.0);

-- Crop Recommendation
INSERT INTO crop_recommendation (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, recommended_crop)
VALUES
(90, 40, 40, 25, 80, 6.5, 200, 'Wheat'),
(50, 60, 40, 30, 70, 6.8, 150, 'Rice');

-- Disease Detection
INSERT INTO disease_detection (image_path, predicted_disease, confidence)
VALUES
('train/Healthy/leaf1.jpg', 'Healthy', 0.98),
('train/Powdery_Mildew/leaf2.jpg', 'Powdery_Mildew', 0.92);
