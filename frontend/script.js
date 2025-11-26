// Minimal JS for AgroSmart
console.log('AgroSmart loaded');

function recommendCrop() {
    const N = parseFloat(document.getElementById('N').value);
    const P = parseFloat(document.getElementById('P').value);
    const K = parseFloat(document.getElementById('K').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const ph = parseFloat(document.getElementById('ph').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    fetch('http://localhost:5000/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ N, P, K, temperature, humidity, ph, rainfall })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('result').textContent = data.recommendation || data.error;
    });
}
