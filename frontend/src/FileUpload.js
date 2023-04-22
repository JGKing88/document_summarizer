import React, {Component} from 'react';
import axios from 'axios';

class FileUpload extends Component {
  
    state = {
 
      // Initially, no file is selected
      onSelectedFile: null
    };
    
    // On file select (from the pop up)
    onFileChange = event => {
    
      // Update the state
      this.setState({ onSelectedFile: event.target.files[0] });
    
    };
    
    // On file upload (click the upload button)
    onFileUpload = () => {
    
      // Create an object of formData
      const formData = new FormData();
    
      // Update the formData object
      formData.append(
        "myFile",
        this.state.onSelectedFile,
        this.state.onSelectedFile.name
      );
    
      // Details of the uploaded file
      console.log(this.state.onSelectedFile);
    
      // Request made to the backend api
      // Send formData object
      axios.post("api/uploadfile", formData);
    };
    
    // File content to be displayed after
    // file upload is complete
    fileData = () => {
    
      if (this.state.onSelectedFile) {
         
        return (
          <div>
            <h2>File Information:</h2>
             
<p>File Name: {this.state.onSelectedFile.name}</p>
 
             
<p>File Type: {this.state.onSelectedFile.type}</p>
 
             
<p>
              Last Modified:{" "}
              {this.state.onSelectedFile.lastModifiedDate.toDateString()}
            </p>
 
          </div>
        );
      } else {
        return (
          <div>
            <br />
            <h4>Choose file before Pressing the Upload button</h4>
          </div>
        );
      }
    };
    
    render() {
    
      return (
        <div>
            <h3>
              File Upload with Axios
            </h3>
            <div>
                <input type="file" onChange={this.onFileChange} />
                <button onClick={this.onFileUpload}>
                  Upload!
                </button>
            </div>
          {this.fileData()}
        </div>
      );
    }
  }
 

export default FileUpload;