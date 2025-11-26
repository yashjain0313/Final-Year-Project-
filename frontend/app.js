// Application Data
const appData = {
  crops_dataset: [
    {"crop": "rice", "N": 56.6, "P": 38.95, "K": 27.02, "pH": 6.3, "rainfall": 446.5, "temperature": 33.0, "humidity": 73.0},
    {"crop": "wheat", "N": 45.2, "P": 35.8, "K": 42.1, "pH": 7.1, "rainfall": 650.3, "temperature": 22.5, "humidity": 65.2},
    {"crop": "cotton", "N": 83.32, "P": 76.93, "K": 40.04, "pH": 8.0, "rainfall": 932.2, "temperature": 34.2, "humidity": 60.2},
    {"crop": "tomato", "N": 39.03, "P": 35.60, "K": 44.13, "pH": 6.2, "rainfall": 787.8, "temperature": 27.3, "humidity": 79.1},
    {"crop": "potato", "N": 28.5, "P": 42.3, "K": 38.7, "pH": 6.5, "rainfall": 580.4, "temperature": 20.8, "humidity": 72.5}
  ],
  market_prices: [
    {"crop": "rice", "price": 2443.26, "trend": "+5.2%", "market": "Delhi"},
    {"crop": "wheat", "price": 2167.25, "trend": "+2.1%", "market": "Mumbai"}, 
    {"crop": "cotton", "price": 5906.01, "trend": "-1.8%", "market": "Bangalore"},
    {"crop": "tomato", "price": 2000.50, "trend": "+8.5%", "market": "Chennai"},
    {"crop": "potato", "price": 1200.75, "trend": "-3.2%", "market": "Kolkata"}
  ],
  weather_data: [
    {"city": "Delhi", "temp_max": 28.3, "temp_min": 20.3, "humidity": 64.0, "rainfall": 21.3, "condition": "Sunny"},
    {"city": "Mumbai", "temp_max": 30.4, "temp_min": 22.4, "humidity": 83.8, "rainfall": 34.0, "condition": "Rainy"},
    {"city": "Bangalore", "temp_max": 27.5, "temp_min": 19.5, "humidity": 73.1, "rainfall": 12.9, "condition": "Partly Cloudy"},
    {"city": "Chennai", "temp_max": 34.4, "temp_min": 26.4, "humidity": 80.7, "rainfall": 29.8, "condition": "Sunny"},
    {"city": "Hyderabad", "temp_max": 32.1, "temp_min": 24.2, "humidity": 70.5, "rainfall": 18.7, "condition": "Sunny"},
    {"city": "Ahmedabad", "temp_max": 35.0, "temp_min": 25.0, "humidity": 60.0, "rainfall": 10.2, "condition": "Sunny"},
    {"city": "Pune", "temp_max": 29.8, "temp_min": 21.3, "humidity": 68.2, "rainfall": 15.4, "condition": "Partly Cloudy"},
    {"city": "Kolkata", "temp_max": 31.5, "temp_min": 24.8, "humidity": 78.9, "rainfall": 27.6, "condition": "Rainy"},
    {"city": "Jaipur", "temp_max": 36.2, "temp_min": 26.1, "humidity": 55.3, "rainfall": 8.9, "condition": "Sunny"},
  {"city": "Lucknow", "temp_max": 33.7, "temp_min": 23.5, "humidity": 65.7, "rainfall": 13.8, "condition": "Sunny"},
  {"city": "Visakhapatnam", "temp_max": 31.2, "temp_min": 25.6, "humidity": 77.4, "rainfall": 22.1, "condition": "Partly Cloudy"}
  ],
  crop_information: [
    {
      "crop": "rice",
      "name": "Rice",
      "season": "Kharif/Rabi",
      "duration": "120-150 days",
      "yield": "40-60 quintals/hectare",
      "description": "Rice is the staple food for billions of people. It requires flooded fields and warm climate.",
      "tips": "Prepare nursery beds, Transplant after 25-30 days, Maintain water level, Apply fertilizers in splits"
    },
    {
      "crop": "wheat", 
      "name": "Wheat",
      "season": "Rabi",
      "duration": "120-150 days", 
      "yield": "35-50 quintals/hectare",
      "description": "Wheat is a major cereal grain and staple food crop grown in temperate regions.",
      "tips": "Sow in October-November, Use certified seeds, Apply irrigation at critical stages, Control weeds timely"
    },
    {
      "crop": "cotton",
      "name": "Cotton", 
      "season": "Kharif",
      "duration": "180-200 days",
      "yield": "15-25 quintals/hectare",
      "description": "Cotton is the most important fiber crop providing raw material to textile industry.",
      "tips": "Deep ploughing, Use Bt cotton varieties, Integrated pest management, Pick cotton when bolls open"
    },
    {
      "crop": "tomato",
      "name": "Tomato",
      "season": "Kharif/Rabi",
      "duration": "90-120 days",
      "yield": "400-600 quintals/hectare",
      "description": "Tomato is a popular vegetable crop rich in vitamins and minerals.",
      "tips": "Use disease-resistant varieties, Provide support to plants, Regular watering, Harvest at proper maturity"
    },
    {
      "crop": "potato",
      "name": "Potato",
      "season": "Rabi",
      "duration": "70-120 days",
      "yield": "200-400 quintals/hectare",
      "description": "Potato is an important food crop and source of carbohydrates.",
      "tips": "Plant in well-drained soil, Earthing up is essential, Control late blight disease, Harvest when plants mature"
    }
  ],
  success_metrics: {
    "farmers_helped": "75,000+",
    "yield_improvement": "18%", 
    "cost_reduction": "22%",
    "water_savings": "30%",
    "revenue_increase": "₹45,000",
    "app_rating": "4.8/5"
  },
  states: [
    "Uttar Pradesh", "Punjab", "Haryana", "Rajasthan", "Maharashtra", 
    "Madhya Pradesh", "Karnataka", "Andhra Pradesh", "Tamil Nadu", 
    "West Bengal", "Bihar", "Gujarat", "Odisha", "Kerala"
  ]
};

// Global Variables
let currentSection = 'home';
let weatherChart = null;
let priceChart = null;
let impactChart = null;
let isVoiceActive = false;

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing app...');
  initializeApp();
  setupDiseaseDetection();
});

function initializeApp() {
  setupNavigation();
  setupForms();
  setupModals();
  setupWeatherDashboard();
  setupMarketPrices();
  setupCropLibrary();
  setupVoiceAssistant();
  populateStateDropdown();
  checkOfflineStatus();
  setupDiseaseDetection();
  // Initialize with home section active
  navigateToSection('home');
  console.log('App initialized successfully');
// Disease Detection Logic
function setupDiseaseDetection() {
  const diseaseForm = document.getElementById('diseaseForm');
  const diseaseImageInput = document.getElementById('diseaseImage');
  const diseasePreview = document.getElementById('diseasePreview');
  const diseaseResult = document.getElementById('diseaseResult');
  const diseaseLabel = document.getElementById('diseaseLabel');
  const diseaseDescription = document.getElementById('diseaseDescription');
  const diseaseInfoPara = document.getElementById('diseaseInfoPara');
  const loadingSpinner = document.getElementById('diseaseLoadingSpinner');

  if (diseaseImageInput) {
    diseaseImageInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function(ev) {
          diseasePreview.innerHTML = `<img src="${ev.target.result}" alt="Leaf Preview" class="disease-img-preview" style="max-width: 300px; border-radius: 8px; box-shadow: 0 2px 8px #ccc;" />`;
        };
        reader.readAsDataURL(file);
      } else {
        diseasePreview.innerHTML = '';
      }
    });
  }

  if (diseaseForm) {
    diseaseForm.addEventListener('submit', function(e) {
      e.preventDefault();
      const file = diseaseImageInput.files[0];
      if (!file) {
        alert('Please upload a leaf image.');
        return;
      }
      diseaseResult.classList.add('hidden');
      loadingSpinner.classList.remove('hidden');
      const formData = new FormData();
      formData.append('image', file);
      fetch('/api/disease', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        loadingSpinner.classList.add('hidden');
        if (data.label) {
          diseaseLabel.textContent = data.label;
          // Disease descriptions database
          const diseaseDescriptions = {
            "Apple - Apple Scab": "Apple scab is a fungal disease caused by Venturia inaequalis. It creates dark, scabby lesions on leaves and fruit, reducing yield and quality. Control by removing fallen leaves and applying fungicides in early spring.",
            "Apple - Black Rot": "Black rot, caused by Botryosphaeria obtusa, affects apples with dark, sunken spots and fruit decay. Prune infected branches and use copper-based fungicides to prevent spread.",
            "Apple - Cedar Apple Rust": "Cedar apple rust is a fungal disease causing yellow-orange spots on apple leaves. Remove nearby cedar trees and apply fungicides at bud break for effective control.",
            "Apple - Healthy": "No visible symptoms of disease. Your apple plant is healthy. Continue regular care and monitoring.",
            "Blueberry - Healthy": "No visible symptoms of disease. Your blueberry plant is healthy. Continue regular care and monitoring.",
            "Cherry (Sour) - Powdery Mildew": "Powdery mildew appears as white powdery spots on cherry leaves and stems. Improve air circulation and use sulfur-based fungicides to manage the disease.",
            "Cherry (Sour) - Healthy": "No visible symptoms of disease. Your cherry plant is healthy. Continue regular care and monitoring.",
            "Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot": "Gray leaf spot, caused by Cercospora zeae-maydis, creates rectangular gray lesions on corn leaves. Rotate crops and use resistant varieties to reduce impact.",
            "Corn (Maize) - Common Rust": "Common rust causes reddish-brown pustules on corn leaves. Remove infected leaves and apply fungicides to control the disease.",
            "Corn (Maize) - Northern Leaf Blight": "Northern leaf blight causes long, gray-green lesions on corn leaves. Rotate crops and use resistant varieties for management.",
            "Corn (Maize) - Healthy": "No visible symptoms of disease. Your corn plant is healthy. Continue regular care and monitoring.",
            "Grape - Black Rot": "Black rot, caused by Guignardia bidwellii, affects grapes with dark, sunken spots and fruit decay. Prune infected branches and use fungicides to prevent spread.",
            "Grape - Esca (Black Measles)": "Esca, or black measles, is a complex grapevine disease causing leaf discoloration and fruit shriveling. Remove infected wood and avoid vine stress.",
            "Grape - Leaf Blight (Isariopsis Leaf Spot)": "Isariopsis leaf spot causes brown, angular spots on grape leaves. Remove affected leaves and apply fungicides for control.",
            "Grape - Healthy": "No visible symptoms of disease. Your grapevine is healthy. Continue regular care and monitoring.",
            "Orange - Citrus Greening (HLB)": "Citrus greening (HLB) is a bacterial disease causing yellow shoots, misshapen fruit, and tree decline. Control psyllid vectors and remove infected trees.",
            "Peach - Bacterial Spot": "Bacterial spot causes water-soaked lesions and leaf drop in peaches. Avoid overhead watering and use copper-based sprays for management.",
            "Peach - Healthy": "No visible symptoms of disease. Your peach tree is healthy. Continue regular care and monitoring.",
            "Pepper (Bell) - Bacterial Spot": "Bacterial spot leads to water-soaked lesions and leaf drop in bell peppers. Avoid overhead watering and use copper-based sprays.",
            "Pepper (Bell) - Healthy": "No visible symptoms of disease. Your bell pepper plant is healthy. Continue regular care and monitoring.",
            "Potato - Early Blight": "Early blight, caused by Alternaria solani, creates concentric rings on potato leaves and stems. Remove infected leaves and rotate crops regularly.",
            "Potato - Late Blight": "Late blight, caused by Phytophthora infestans, causes water-soaked lesions and rapid plant decay in potatoes. Remove infected plants and apply fungicides promptly.",
            "Potato - Healthy": "No visible symptoms of disease. Your potato plant is healthy. Continue regular care and monitoring.",
            "Raspberry - Healthy": "No visible symptoms of disease. Your raspberry plant is healthy. Continue regular care and monitoring.",
            "Soybean - Healthy": "No visible symptoms of disease. Your soybean plant is healthy. Continue regular care and monitoring.",
            "Squash - Powdery Mildew": "Powdery mildew appears as white powdery spots on squash leaves and stems. Improve air circulation and use sulfur-based fungicides.",
            "Strawberry - Leaf Scorch": "Leaf scorch causes reddish-brown spots and leaf drying in strawberries. Remove affected leaves and use resistant varieties.",
            "Strawberry - Healthy": "No visible symptoms of disease. Your strawberry plant is healthy. Continue regular care and monitoring.",
            "Tomato - Bacterial Spot": "Bacterial spot leads to water-soaked lesions and leaf drop in tomatoes. Avoid overhead watering and use copper-based sprays.",
            "Tomato - Early Blight": "Early blight, caused by Alternaria solani, creates concentric rings on tomato leaves and fruit. Remove infected leaves and rotate crops regularly.",
            "Tomato - Late Blight": "Late blight, caused by Phytophthora infestans, causes water-soaked lesions and rapid plant decay in tomatoes. Remove infected plants and apply fungicides promptly.",
            "Tomato - Leaf Mold": "Leaf mold causes yellow spots and fuzzy growth on tomato leaves. Improve air circulation and use fungicides for control.",
            "Tomato - Septoria Leaf Spot": "Septoria leaf spot causes small, dark spots with yellow halos on tomato leaves. Remove affected leaves and use fungicides.",
            "Tomato - Spider Mites / Two-Spotted Spider Mite": "Spider mites cause stippling and webbing on tomato leaves. Spray with water and use miticides if needed.",
            "Tomato - Target Spot": "Target spot causes circular lesions with concentric rings on tomato leaves. Remove infected leaves and apply fungicides.",
            "Tomato - Yellow Leaf Curl Virus": "Yellow leaf curl virus causes leaf curling and yellowing in tomatoes. Control whiteflies and use virus-resistant varieties.",
            "Tomato - Mosaic Virus": "Mosaic virus causes mottled, distorted leaves in tomatoes. Remove infected plants and control aphids.",
            "Tomato - Healthy": "No visible symptoms of disease. Your tomato plant is healthy. Continue regular care and monitoring."
          };
          // Normalize label for robust mapping
          const normalizedLabel = (data.label || '').trim();
          const description = diseaseDescriptions[normalizedLabel];
          console.log('Disease label:', normalizedLabel, '| Description:', description);
          diseaseDescription.innerHTML = `<p style='margin:0;'>${description ? description : 'No description available for this disease.'}</p>`;
          diseaseInfoPara.textContent = '';
          diseaseResult.classList.remove('hidden');
        } else {
          diseaseLabel.textContent = 'No disease detected.';
          diseaseDescription.textContent = '';
          diseaseInfoPara.textContent = '';
          diseaseResult.classList.remove('hidden');
        }
      })
      .catch(err => {
        loadingSpinner.classList.add('hidden');
        diseaseLabel.textContent = 'Error detecting disease.';
        diseaseDescription.textContent = '';
        diseaseResult.classList.remove('hidden');
        console.error('Disease detection error:', err);
      });
    });
  }
}
}

// Navigation System
function setupNavigation() {
  console.log('Setting up navigation...');
  
  // Use event delegation for better compatibility
  document.addEventListener('click', function(e) {
    // Handle navigation links
    if (e.target.matches('[data-section]') || e.target.closest('[data-section]')) {
      e.preventDefault();
      const target = e.target.matches('[data-section]') ? e.target : e.target.closest('[data-section]');
      const section = target.getAttribute('data-section');
      console.log('Navigation clicked:', section);
      if (section) {
        navigateToSection(section);
      }
      return false;
    }
  });
  
  console.log('Navigation setup complete');
}

function navigateToSection(sectionId) {
  console.log('Navigating to section:', sectionId);
  
  // Hide all sections
  const sections = document.querySelectorAll('.section');
  sections.forEach(section => {
    section.classList.remove('active');
  });

  // Show target section
  const targetSection = document.getElementById(sectionId);
  if (targetSection) {
    targetSection.classList.add('active');
    currentSection = sectionId;
    console.log('Section activated:', sectionId);

    // Update navigation states
    updateNavigationState(sectionId);

    // Initialize section-specific functionality
    setTimeout(() => {
      if (sectionId === 'analytics') {
        setupAnalytics();
      } else if (sectionId === 'market') {
        setupMarketPrices();
      } else if (sectionId === 'learn') {
        setupCropLibrary();
      } else if (sectionId === 'weather') {
        setupWeatherDashboard();
      }
    }, 100);
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } else {
    console.error('Section not found:', sectionId);
  }
}

function updateNavigationState(activeSection) {
  console.log('Updating navigation state for:', activeSection);
  
  // Update desktop nav
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('data-section') === activeSection) {
      link.classList.add('active');
    }
  });

  // Update mobile nav
  const bottomNavItems = document.querySelectorAll('.bottom-nav-item');
  bottomNavItems.forEach(item => {
    item.classList.remove('active');
    if (item.getAttribute('data-section') === activeSection) {
      item.classList.add('active');
    }
  });
}

// Form Setup and Validation
function setupForms() {
  console.log('Setting up forms...');
  
  const cropForm = document.getElementById('cropForm');
  if (cropForm) {
    cropForm.addEventListener('submit', handleCropRecommendation);
    console.log('Crop form event listener added');
  } else {
    console.error('Crop form not found');
  }
}

function populateStateDropdown() {
  console.log('Populating state dropdown...');
  
  const stateSelect = document.getElementById('state');
  if (stateSelect) {
    // Clear existing options
    stateSelect.innerHTML = '<option value="">Select State</option>';
    
    // Add state options
    appData.states.forEach(state => {
      const option = document.createElement('option');
      option.value = state;
      option.textContent = state;
      stateSelect.appendChild(option);
    });
    
    console.log('State dropdown populated with', appData.states.length, 'states');
  } else {
    console.error('State dropdown not found');
  }
}

function handleCropRecommendation(e) {
  e.preventDefault();
  console.log('Form submitted, processing crop recommendation...');
  
  // Get form values
  const nitrogenEl = document.getElementById('nitrogen');
  const phosphorusEl = document.getElementById('phosphorus');
  const potassiumEl = document.getElementById('potassium');
  const phEl = document.getElementById('ph');
  const temperatureEl = document.getElementById('temperature');
  const humidityEl = document.getElementById('humidity');
  const rainfallEl = document.getElementById('rainfall');
  const stateEl = document.getElementById('state');
  
  if (!nitrogenEl || !phosphorusEl || !potassiumEl || !phEl || !temperatureEl || !humidityEl || !rainfallEl || !stateEl) {
    console.error('Form elements not found');
    alert('Form elements not found. Please refresh the page.');
    return;
  }
  
  const formData = {
    N: parseFloat(nitrogenEl.value),
    P: parseFloat(phosphorusEl.value),
    K: parseFloat(potassiumEl.value),
    pH: parseFloat(phEl.value),
    temperature: parseFloat(temperatureEl.value),
    humidity: parseFloat(humidityEl.value),
    rainfall: parseFloat(rainfallEl.value),
    state: stateEl.value
  };

  console.log('Form data:', formData);

  // Validate form data
  if (!formData.N || !formData.P || !formData.K || !formData.pH || !formData.temperature || !formData.humidity || !formData.rainfall || !formData.state) {
    alert('Please fill in all fields');
    return;
  }

  // Show loading state
  const submitBtn = e.target.querySelector('button[type="submit"]');
  const spinner = document.getElementById('loadingSpinner');
  if (submitBtn) {
    submitBtn.disabled = true;
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    spinner && spinner.classList.remove('hidden');
    // Prepare data for backend
    const payload = {
      N: formData.N,
      P: formData.P,
      K: formData.K,
      ph: formData.pH,
      temperature: formData.temperature,
      humidity: formData.humidity,
      rainfall: formData.rainfall
    };
    fetch('/api/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
      if (data.recommendation) {
        // If backend returns a string, display it directly
        displayRecommendations([{ name: data.recommendation }]);
      } else {
        alert('No recommendation found. Please check your input.');
      }
    })
    .catch(err => {
      console.error('Error fetching recommendation:', err);
      alert('Error fetching recommendation. Please try again.');
    })
    .finally(() => {
      submitBtn.disabled = false;
      submitBtn.innerHTML = originalText;
      spinner && spinner.classList.add('hidden');
    });
  }
}

function getCropRecommendations(userInput) {
  console.log('Calculating crop recommendations for:', userInput);
  
  const scores = appData.crops_dataset.map(crop => {
    let score = 0;

    // Calculate similarity scores for each parameter (lower difference = higher score)
    const nDiff = Math.abs(crop.N - userInput.N) / Math.max(crop.N, userInput.N, 1);
    const pDiff = Math.abs(crop.P - userInput.P) / Math.max(crop.P, userInput.P, 1);
    const kDiff = Math.abs(crop.K - userInput.K) / Math.max(crop.K, userInput.K, 1);
    const phDiff = Math.abs(crop.pH - userInput.pH) / Math.max(crop.pH, userInput.pH, 1);
    const tempDiff = Math.abs(crop.temperature - userInput.temperature) / Math.max(crop.temperature, userInput.temperature, 1);
    const humDiff = Math.abs(crop.humidity - userInput.humidity) / Math.max(crop.humidity, userInput.humidity, 1);
    const rainDiff = Math.abs(crop.rainfall - userInput.rainfall) / Math.max(crop.rainfall, userInput.rainfall, 1);

    // Weight the factors (lower difference = higher score)
    score += (1 - Math.min(nDiff, 1)) * 0.2;
    score += (1 - Math.min(pDiff, 1)) * 0.2;
    score += (1 - Math.min(kDiff, 1)) * 0.15;
    score += (1 - Math.min(phDiff, 1)) * 0.15;
    score += (1 - Math.min(tempDiff, 1)) * 0.15;
    score += (1 - Math.min(humDiff, 1)) * 0.10;
    score += (1 - Math.min(rainDiff, 1)) * 0.05;

    return {
      crop: crop.crop,
      score: Math.max(0, score),
      confidence: Math.round(Math.max(0, score) * 100)
    };
  });

  // Sort by score and return top 3
  const recommendations = scores
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(item => ({
      ...item,
      ...appData.crop_information.find(info => info.crop === item.crop),
      marketPrice: appData.market_prices.find(price => price.crop === item.crop)
    }));
    
  console.log('Generated recommendations:', recommendations);
  return recommendations;
}

function displayRecommendations(recommendations) {
  console.log('Displaying recommendations...');
  
  const resultsContainer = document.getElementById('recommendationsResults');
  const gridContainer = document.getElementById('recommendationsGrid');

  if (!resultsContainer || !gridContainer) {
    console.error('Results containers not found');
    return;
  }

  gridContainer.innerHTML = '';

  recommendations.forEach((rec, index) => {
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    card.innerHTML = `
      <div class="recommendation-header">
        <div class="crop-name">${rec.name || rec.crop}</div>
      </div>
      <p class="crop-description">${rec.description || 'No description available.'}</p>
      <div class="crop-actions">
        <button class="btn btn--primary btn--sm" onclick="showCropDetails('${rec.crop}')">
          <i class="fas fa-info-circle"></i> Learn More
        </button>
        <button class="btn btn--outline btn--sm" onclick="navigateToSection('market')">
          <i class="fas fa-chart-line"></i> View Prices
        </button>
      </div>
    `;
    gridContainer.appendChild(card);
  });

  resultsContainer.classList.remove('hidden');
  resultsContainer.scrollIntoView({ behavior: 'smooth' });
  console.log('Recommendations displayed successfully');
}

// Weather Dashboard
function setupWeatherDashboard() {
  console.log('Setting up weather dashboard...');
  
  const citySelector = document.getElementById('citySelector');
  if (citySelector) {
    // Clear existing options
    citySelector.innerHTML = '<option value="">Select City</option>';
    // Add city options from appData.weather_data
    appData.weather_data.forEach(cityObj => {
      const option = document.createElement('option');
      option.value = cityObj.city;
      option.textContent = cityObj.city;
      citySelector.appendChild(option);
    });
    // Remove existing listeners
    citySelector.removeEventListener('change', handleCitySelection);
    citySelector.addEventListener('change', handleCitySelection);
    console.log('City selector event listener added and populated with cities');
  } else {
    console.error('City selector not found');
  }
}

function handleCitySelection(e) {
  const selectedCity = e.target.value;
  console.log('City selected:', selectedCity);
  if (selectedCity) {
    // Fetch live weather data from backend
    fetch(`/api/weather?city=${encodeURIComponent(selectedCity)}`)
      .then(response => response.json())
      .then(weatherData => {
        if (weatherData.error) {
          console.error('Weather API error:', weatherData.error);
          alert('Weather data not available for this city.');
          return;
        }
        // Pass backend data to display and chart functions
        displayWeatherData(weatherData);
        setTimeout(() => createWeatherChart(weatherData), 100);
        generateWeatherAdvisories(weatherData);
      })
      .catch(err => {
        console.error('Error fetching weather:', err);
        alert('Error fetching weather data.');
      });
  }
}

function displayWeatherData(data) {
  console.log('Displaying weather data for:', data.city);
  const weatherContent = document.getElementById('weatherContent');
  const currentWeather = document.getElementById('currentWeather');
  if (!weatherContent || !currentWeather) {
    console.error('Weather elements not found');
    return;
  }
  // Use backend data structure
  const iconCode = data.current && data.current.weathercode !== undefined ? data.current.weathercode : null;
  const weatherIcon = getWeatherIcon(iconCode);
  const temp = data.current && data.current.temperature ? data.current.temperature : 'N/A';
  // Use backend-provided humidity (average for today)
  const humidity = (typeof data.humidity !== 'undefined' && data.humidity !== null) ? data.humidity : (data.current && data.current.relative_humidity ? data.current.relative_humidity : 'N/A');
  // Daily min/max
  const minTemp = data.daily && data.daily.temperature_2m_min ? data.daily.temperature_2m_min[0] : 'N/A';
  const maxTemp = data.daily && data.daily.temperature_2m_max ? data.daily.temperature_2m_max[0] : 'N/A';
  currentWeather.innerHTML = `
    <div class="weather-header">
      <div class="weather-city">${data.city}</div>
      <div class="weather-icon">${weatherIcon}</div>
    </div>
    <div class="weather-stats">
      <div class="weather-stat">
        <div class="weather-value">${maxTemp}°C</div>
        <div class="weather-label">Max Temp</div>
      </div>
      <div class="weather-stat">
        <div class="weather-value">${minTemp}°C</div>
        <div class="weather-label">Min Temp</div>
      </div>
      <div class="weather-stat">
        <div class="weather-value">${humidity}%</div>
        <div class="weather-label">Humidity</div>
      </div>
      <div class="weather-stat">
        <div class="weather-value">${typeof data.rainfall !== 'undefined' && data.rainfall !== null ? data.rainfall : 'N/A'} mm</div>
        <div class="weather-label">Rainfall</div>
      </div>
    </div>
  `;

  weatherContent.classList.remove('hidden');
  console.log('Weather data displayed successfully');
}

function getWeatherIcon(condition) {
  const icons = {
    'Sunny': '<i class="fas fa-sun"></i>',
    'Rainy': '<i class="fas fa-cloud-rain"></i>',
    'Partly Cloudy': '<i class="fas fa-cloud-sun"></i>',
    'Cloudy': '<i class="fas fa-cloud"></i>'
  };
  return icons[condition] || '<i class="fas fa-sun"></i>';
}

function createWeatherChart(city) {
  // Accept backend weather data object
  const cityName = city.city || city;
  console.log('Creating weather chart for:', cityName);
  const ctx = document.getElementById('weatherChart');
  if (!ctx) {
    console.error('Weather chart canvas not found');
    return;
  }
  if (weatherChart) {
    weatherChart.destroy();
  }
  // Use backend data for 7-day forecast
  const daily = city.daily || {};
  const maxTemp = daily.temperature_2m_max || [];
  const minTemp = daily.temperature_2m_min || [];
  weatherChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
      datasets: [
        {
          label: 'Max Temperature (°C)',
          data: maxTemp,
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          tension: 0.4
        },
        {
          label: 'Min Temperature (°C)',
          data: minTemp,
          borderColor: '#FFC185',
          backgroundColor: 'rgba(255, 193, 133, 0.1)',
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: 'Temperature (°C)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: `7-Day Weather Forecast - ${cityName}`
        }
      }
    }
  });
  console.log('Weather chart created successfully');
}

function generateForecastData(city) {
  // No longer needed: chart uses backend data
  return { maxTemp: [], minTemp: [] };
}

function generateWeatherAdvisories(weatherData) {
  const advisoryList = document.getElementById('advisoryList');
  if (!advisoryList) {
    console.error('Advisory list not found');
    return;
  }
  
  const advisories = [];

  if (weatherData.rainfall > 30) {
    advisories.push({
      icon: 'fas fa-cloud-rain',
      title: 'Heavy Rainfall Alert',
      text: 'Ensure proper drainage in fields. Delay harvesting if crops are ready.'
    });
  }

  if (weatherData.temp_max > 35) {
    advisories.push({
      icon: 'fas fa-thermometer-three-quarters',
      title: 'High Temperature Warning',
      text: 'Increase irrigation frequency. Provide shade for livestock.'
    });
  }

  if (weatherData.humidity > 80) {
    advisories.push({
      icon: 'fas fa-eye-dropper',
      title: 'High Humidity Alert',
      text: 'Monitor crops for fungal diseases. Ensure good air circulation.'
    });
  }

  if (advisories.length === 0) {
    advisories.push({
      icon: 'fas fa-leaf',
      title: 'Favorable Conditions',
      text: 'Weather conditions are favorable for most farming activities.'
    });
  }

  advisoryList.innerHTML = advisories.map(advisory => `
    <div class="advisory-item">
      <div class="advisory-icon">
        <i class="${advisory.icon}"></i>
      </div>
      <div class="advisory-content">
        <div class="advisory-title">${advisory.title}</div>
        <div class="advisory-text">${advisory.text}</div>
      </div>
    </div>
  `).join('');
}

// Market Prices
function setupMarketPrices() {
  console.log('Setting up market prices...');
  displayMarketPrices();
  setTimeout(() => createPriceChart(), 100);
}

function displayMarketPrices() {
  const priceTableBody = document.getElementById('priceTableBody');
  if (!priceTableBody) {
    console.error('Price table body not found');
    return;
  }

  priceTableBody.innerHTML = appData.market_prices.map(price => `
    <tr>
      <td>
        <div class="flex items-center gap-8">
          <i class="fas fa-seedling text-success"></i>
          <span style="text-transform: capitalize; font-weight: 500;">${price.crop}</span>
        </div>
      </td>
      <td style="font-weight: 600;">₹${price.price.toLocaleString()}</td>
      <td>
        <span class="${price.trend.startsWith('+') ? 'trend-positive' : 'trend-negative'}">
          ${price.trend}
        </span>
      </td>
      <td>${price.market}</td>
      <td>
        <button class="btn btn--sm btn--outline" onclick="showPriceAnalysis('${price.crop}')">
          <i class="fas fa-chart-line"></i> Analyze
        </button>
      </td>
    </tr>
  `).join('');
  
  console.log('Market prices displayed successfully');
}

function createPriceChart() {
  console.log('Creating price chart...');
  
  const ctx = document.getElementById('priceChart');
  if (!ctx) {
    console.error('Price chart canvas not found');
    return;
  }

  if (priceChart) {
    priceChart.destroy();
  }

  const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'];

  priceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: appData.market_prices.map(item => item.crop.charAt(0).toUpperCase() + item.crop.slice(1)),
      datasets: [{
        label: 'Price (₹/Quintal)',
        data: appData.market_prices.map(item => item.price),
        backgroundColor: chartColors,
        borderColor: chartColors,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Price (₹/Quintal)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Current Market Prices by Crop'
        },
        legend: {
          display: false
        }
      }
    }
  });
  
  console.log('Price chart created successfully');
}

function showPriceAnalysis(crop) {
  const priceData = appData.market_prices.find(item => item.crop === crop);
  if (priceData) {
    alert(`Price Analysis for ${crop.charAt(0).toUpperCase() + crop.slice(1)}:\n\nCurrent Price: ₹${priceData.price.toLocaleString()}/Quintal\nTrend: ${priceData.trend}\nMarket: ${priceData.market}\n\nRecommendation: ${priceData.trend.startsWith('+') ? 'Good time to sell' : 'Consider holding for better prices'}`);
  }
}

// Crop Library
function setupCropLibrary() {
  console.log('Setting up crop library...');
  displayCropLibrary();
  setupCropSearch();
}

function displayCropLibrary(searchTerm = '') {
  const cropLibrary = document.getElementById('cropLibrary');
  if (!cropLibrary) {
    console.error('Crop library not found');
    return;
  }

  const filteredCrops = appData.crop_information.filter(crop =>
    crop.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    crop.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  cropLibrary.innerHTML = filteredCrops.map(crop => `
    <div class="crop-card" onclick="showCropDetails('${crop.crop}')">
      <div class="crop-card-header">
        <div class="crop-card-title">${crop.name}</div>
        <div class="crop-season">${crop.season}</div>
      </div>
      <div class="crop-card-content">
        <div class="crop-description">${crop.description}</div>
        <div class="crop-info-grid">
          <div class="crop-info-item">
            <span class="crop-info-label">Duration:</span>
            <span class="crop-info-value">${crop.duration}</span>
          </div>
          <div class="crop-info-item">
            <span class="crop-info-label">Yield:</span>
            <span class="crop-info-value">${crop.yield}</span>
          </div>
        </div>
      </div>
    </div>
  `).join('');
  
  console.log('Crop library displayed with', filteredCrops.length, 'crops');
}

function setupCropSearch() {
  const searchInput = document.getElementById('cropSearch');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      displayCropLibrary(e.target.value);
    });
    console.log('Crop search setup complete');
  }
}

function showCropDetails(cropName) {
  const crop = appData.crop_information.find(item => item.crop === cropName);
  const marketPrice = appData.market_prices.find(price => price.crop === cropName);
  
  if (!crop) {
    console.error('Crop not found:', cropName);
    return;
  }

  const modal = document.getElementById('cropModal');
  const modalTitle = document.getElementById('cropModalTitle');
  const modalBody = document.getElementById('cropModalBody');

  if (!modal || !modalTitle || !modalBody) {
    console.error('Crop modal elements not found');
    return;
  }

  modalTitle.textContent = crop.name + ' - Detailed Guide';
  modalBody.innerHTML = `
    <div class="crop-detail-content">
      <div class="crop-detail-header">
        <div class="crop-detail-badge">${crop.season}</div>
        ${marketPrice ? `<div class="crop-price">₹${marketPrice.price.toLocaleString()}/Quintal</div>` : ''}
      </div>
      
      <div class="crop-detail-section">
        <h4>Description</h4>
        <p>${crop.description}</p>
      </div>
      
      <div class="crop-detail-grid">
        <div class="crop-detail-item">
          <strong>Growing Season:</strong> ${crop.season}
        </div>
        <div class="crop-detail-item">
          <strong>Duration:</strong> ${crop.duration}
        </div>
        <div class="crop-detail-item">
          <strong>Expected Yield:</strong> ${crop.yield}
        </div>
        ${marketPrice ? `
        <div class="crop-detail-item">
          <strong>Market Price:</strong> ₹${marketPrice.price.toLocaleString()}/Quintal
        </div>
        <div class="crop-detail-item">
          <strong>Price Trend:</strong> <span class="${marketPrice.trend.startsWith('+') ? 'text-success' : 'text-error'}">${marketPrice.trend}</span>
        </div>
        <div class="crop-detail-item">
          <strong>Market:</strong> ${marketPrice.market}
        </div>
        ` : ''}
      </div>
      
      <div class="crop-detail-section">
        <h4>Cultivation Tips</h4>
        <div class="cultivation-tips">
          ${crop.tips.split(', ').map(tip => `
            <div class="tip-item">
              <i class="fas fa-check-circle text-success"></i>
              <span>${tip}</span>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;

  // Add custom styles for crop detail modal
  if (!document.getElementById('cropDetailStyles')) {
    const style = document.createElement('style');
    style.id = 'cropDetailStyles';
    style.textContent = `
      .crop-detail-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-20);
      }
      .crop-detail-badge {
        background: var(--color-agriculture-light);
        color: var(--color-agriculture-dark);
        padding: var(--space-6) var(--space-12);
        border-radius: var(--radius-full);
        font-size: var(--font-size-sm);
        font-weight: var(--font-weight-medium);
      }
      .crop-price {
        font-size: var(--font-size-xl);
        font-weight: var(--font-weight-bold);
        color: var(--color-agriculture-primary);
      }
      .crop-detail-section {
        margin-bottom: var(--space-24);
      }
      .crop-detail-section h4 {
        color: var(--color-agriculture-primary);
        margin-bottom: var(--space-12);
        border-bottom: 2px solid var(--color-agriculture-light);
        padding-bottom: var(--space-8);
      }
      .crop-detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--space-12);
        margin-bottom: var(--space-20);
      }
      .crop-detail-item {
        padding: var(--space-12);
        background: var(--color-bg-3);
        border-radius: var(--radius-base);
      }
      .cultivation-tips {
        display: grid;
        gap: var(--space-8);
      }
      .tip-item {
        display: flex;
        align-items: flex-start;
        gap: var(--space-8);
        padding: var(--space-8);
        background: var(--color-surface);
        border-radius: var(--radius-base);
        border-left: 3px solid var(--color-agriculture-primary);
      }
      .tip-item i {
        margin-top: var(--space-2);
      }
    `;
    document.head.appendChild(style);
  }

  showModal('cropModal');
}

// Analytics Dashboard
function setupAnalytics() {
  console.log('Setting up analytics...');
  createImpactChart();
}

function createImpactChart() {
  console.log('Creating impact chart...');
  
  const ctx = document.getElementById('impactChart');
  if (!ctx) {
    console.error('Impact chart canvas not found');
    return;
  }

  if (impactChart) {
    impactChart.destroy();
  }

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'];
  const farmersData = [1000, 3500, 8200, 15600, 28300, 42100, 58900, 68200, 75000];
  const revenueData = [50, 180, 420, 890, 1650, 2380, 3200, 3950, 4500];

  impactChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        {
          label: 'Farmers Helped',
          data: farmersData,
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          tension: 0.4,
          yAxisID: 'y'
        },
        {
          label: 'Revenue Generated (₹ Lakhs)',
          data: revenueData,
          borderColor: '#FFC185',
          backgroundColor: 'rgba(255, 193, 133, 0.1)',
          tension: 0.4,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      scales: {
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Number of Farmers'
          }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: {
            display: true,
            text: 'Revenue (₹ Lakhs)'
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: 'Platform Growth Over Time'
        }
      }
    }
  });
  
  console.log('Impact chart created successfully');
}

// Voice Assistant
function setupVoiceAssistant() {
  console.log('Setting up voice assistant...');
  
  const voiceBtn = document.getElementById('voiceBtn');
  if (voiceBtn) {
    voiceBtn.addEventListener('click', () => showModal('voiceModal'));
  }

  const voiceCircle = document.getElementById('voiceCircle');
  if (voiceCircle) {
    voiceCircle.addEventListener('click', toggleVoiceRecording);
  }
}

function toggleVoiceRecording() {
  const voiceCircle = document.getElementById('voiceCircle');
  const voiceStatus = document.getElementById('voiceStatus');
  const voiceCommand = document.getElementById('voiceCommand');

  if (!isVoiceActive) {
    isVoiceActive = true;
    voiceCircle.classList.add('active');
    voiceStatus.textContent = 'Listening...';
    voiceCommand.textContent = 'Speak now';

    setTimeout(() => {
      const responses = [
        'Based on your location, I recommend growing Rice or Wheat.',
        'Current weather conditions are favorable for planting.',
        'Market prices for Tomato are trending upward.',
        'Your soil conditions are suitable for Cotton cultivation.'
      ];
      
      const response = responses[Math.floor(Math.random() * responses.length)];
      voiceStatus.textContent = 'Response:';
      voiceCommand.textContent = response;
      
      voiceCircle.classList.remove('active');
      isVoiceActive = false;
      
      setTimeout(() => {
        voiceStatus.textContent = 'Click to start voice input';
        voiceCommand.textContent = 'Try saying: "What crops should I grow?"';
      }, 5000);
    }, 3000);
  }
}

// Modal System
function setupModals() {
  console.log('Setting up modals...');
  
  const voiceModalClose = document.getElementById('voiceModalClose');
  const voiceModalOverlay = document.getElementById('voiceModalOverlay');
  
  if (voiceModalClose) voiceModalClose.addEventListener('click', () => hideModal('voiceModal'));
  if (voiceModalOverlay) voiceModalOverlay.addEventListener('click', () => hideModal('voiceModal'));

  const cropModalClose = document.getElementById('cropModalClose');
  const cropModalOverlay = document.getElementById('cropModalOverlay');
  
  if (cropModalClose) cropModalClose.addEventListener('click', () => hideModal('cropModal'));
  if (cropModalOverlay) cropModalOverlay.addEventListener('click', () => hideModal('cropModal'));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      hideModal('voiceModal');
      hideModal('cropModal');
    }
  });
}

function showModal(modalId) {
  console.log('Showing modal:', modalId);
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
}

function hideModal(modalId) {
  console.log('Hiding modal:', modalId);
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('hidden');
    document.body.style.overflow = '';
    
    if (modalId === 'voiceModal') {
      isVoiceActive = false;
      const voiceCircle = document.getElementById('voiceCircle');
      const voiceStatus = document.getElementById('voiceStatus');
      const voiceCommand = document.getElementById('voiceCommand');
      
      if (voiceCircle) voiceCircle.classList.remove('active');
      if (voiceStatus) voiceStatus.textContent = 'Click to start voice input';
      if (voiceCommand) voiceCommand.textContent = 'Try saying: "What crops should I grow?"';
    }
  }
}

// Offline Status
function checkOfflineStatus() {
  const offlineIndicator = document.getElementById('offlineIndicator');
  
  function updateOnlineStatus() {
    if (navigator.onLine) {
      if (offlineIndicator) offlineIndicator.classList.add('hidden');
    } else {
      if (offlineIndicator) offlineIndicator.classList.remove('hidden');
    }
  }

  window.addEventListener('online', updateOnlineStatus);
  window.addEventListener('offline', updateOnlineStatus);
  updateOnlineStatus();
}

// Language Selector
const languageSelector = document.getElementById('languageSelector');
if (languageSelector) {
  languageSelector.addEventListener('change', (e) => {
    const selectedLanguage = e.target.value;
    console.log(`Language changed to: ${selectedLanguage}`);
    
    if (selectedLanguage !== 'en') {
      alert('Language switching feature would be implemented here for production use.');
    }
  });
}

// Export functions for global use
window.navigateToSection = navigateToSection;
window.showCropDetails = showCropDetails;
window.showPriceAnalysis = showPriceAnalysis;
window.showModal = showModal;
window.hideModal = hideModal;