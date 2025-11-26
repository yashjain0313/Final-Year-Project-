-- Create schema
CREATE SCHEMA IF NOT EXISTS `AgroSmart`;

-- Users table
CREATE TABLE IF NOT EXISTS `AgroSmart`.`users` (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role ENUM('admin','farmer','guest') DEFAULT 'farmer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Crops table
CREATE TABLE IF NOT EXISTS `AgroSmart`.`crops` (
    crop_id INT AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR(100) NOT NULL,
    type VARCHAR(50),
    optimal_n INT,
    optimal_p INT,
    optimal_k INT,
    optimal_ph FLOAT,
    optimal_temperature FLOAT,
    optimal_humidity FLOAT,
    optimal_rainfall FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Crop Recommendation table
CREATE TABLE IF NOT EXISTS `AgroSmart`.`crop_recommendation` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nitrogen FLOAT,
    phosphorus FLOAT,
    potassium FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    ph FLOAT,
    rainfall FLOAT,
    recommended_crop VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Disease Detection table
CREATE TABLE IF NOT EXISTS `AgroSmart`.`disease_detection` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    predicted_disease VARCHAR(50),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crop Yield table
CREATE TABLE IF NOT EXISTS `AgroSmart`.`crop_yield` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    crop_name VARCHAR(50),
    area FLOAT,
    production FLOAT,
    yield FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
