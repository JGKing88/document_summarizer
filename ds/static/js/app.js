document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const pdfFile = document.getElementById('pdf_file').files[0];
    const userInformation = document.getElementById('user_information').value;
    const summaryDetails = document.getElementById('summary_details').value;
    
    if (!pdfFile) {
        alert('Please select a PDF file.');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', pdfFile);
    formData.append('user_information', userInformation);
    formData.append('summary_details', summaryDetails);

    // Show the loading icon
    document.getElementById('loading').style.display = 'block';

    const response = await fetch('/upload_pdf', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Hide the loading icon
    document.getElementById('loading').style.display = 'none';

    displaySummary(result.summary);
});

function displaySummary(summary) {
    const summaryContainer = document.getElementById('summary-container');
    summaryContainer.innerHTML = summary.replace(/\n/g, '<br>');
    
    // Show the summary card
    document.getElementById('summary-card').style.display = 'block';
}
