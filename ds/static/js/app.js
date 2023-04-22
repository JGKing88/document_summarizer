document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const pdfFile = document.getElementById('pdf_file').files[0];
    const userInformation = document.getElementById('user_information').value;
    
    if (!pdfFile) {
        alert('Please select a PDF file.');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', pdfFile);
    formData.append('user_information', userInformation);

    const response = await fetch('/upload_pdf', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    displaySummary(result.summary);
});

function displaySummary(summary) {
    const summaryContainer = document.getElementById('summary');
    summaryContainer.innerHTML = `
        <pre>${summary}</pre>
    `;
    
    // Show the summary card
    document.getElementById('summary-card').style.display = 'block';
}
