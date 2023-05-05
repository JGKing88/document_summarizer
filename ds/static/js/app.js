document.getElementById('pdfUpload').addEventListener('change', function (event) {
    const fileName = event.target.files[0].name;
    document.getElementById('pdfUploadLabel').textContent = `Selected: ${fileName}`;
});

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const pdfFile = document.getElementById('pdfUpload').files[0];
    const userInformation = document.getElementById('user_information').value;
    const summaryDetails = document.getElementById('summary_details').value;
    const extras = document.getElementById('extras').value;
    
    if (!pdfFile) {
        alert('Please select a PDF file.');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', pdfFile);
    formData.append('user_information', userInformation);
    formData.append('summary_details', summaryDetails);
    formData.append('extras', extras);

    // Show the loading icon
    document.getElementById('loading').style.display = 'block';

    const response = await fetch('/upload_pdf', {
        method: 'POST',
        body: formData
    });

    
    const jsonResult = await response.json();
    var result = JSON.parse(jsonResult.summary);
    
    // Hide the loading icon
    document.getElementById('loading').style.display = 'none';

    displaySummary(result.summary);
    const parentElement = document.getElementById('feature-container');
    if (result.hasOwnProperty("features")) {
        createList(result.features, parentElement);
        parentElement.innerHTML += " \n <div class=\"copytext-container\"> <button class=\"copytext\" onclick=\"copyFuncFeat()\"><i class=\"fa fa-clone\"></i></button> </div>"
    };
});

function displaySummary(summary) {
    const summaryContainer = document.getElementById('summary-container');
    summaryContainer.innerHTML = summary.replace(/\n/g, '<br>');
    summaryContainer.innerHTML += " \n <div class=\"copytext-container\"> <button class=\"copytext\" onclick=\"copyFuncSum()\"><i class=\"fa fa-clone\"></i></button> </div>"
    // Show the summary card
    document.getElementById('summary-card').style.display = 'block';
}

function createList(obj, parentElement) {
    const ul = document.createElement('ul');
  
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const li = document.createElement('li');
        li.textContent = key + ': ';
  
        if (typeof obj[key] === 'object' && obj[key] !== null) {
          createList(obj[key], li);
        } else {
          li.textContent += obj[key];
        }
  
        ul.appendChild(li);
      }
    }
  
    parentElement.appendChild(ul);
  }


function copyFuncSum(){
    var copyText = document.getElementById('summary-container').innerText;
    navigator.clipboard.writeText(copyText); 
}

function copyFuncFeat(){
    var copyText = document.getElementById('feature-container').innerText;
    navigator.clipboard.writeText(copyText); 
}