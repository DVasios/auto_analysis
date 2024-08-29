// Variables
const base_url = window.location.href;

// API Calls
async function get_profile(formData) 
{
    // URL
    const test_url = base_url + '/profile';

    return await fetch(test_url, {
        'method' : 'POST',
        'body' : formData
    });
}


$(function () {

    // // On analyze
    // $('#file-form').on('submit', function(e) {
    //     e.preventDefault(); 

    //     var fileInput = $('#fileInput')[0].files[0]; // Get the first file

    //     var formData = new FormData(this);

    //     get_profile(formData)
    //         .then(res => {

    //             return res.json()
    //         })
    //         .then(data => {

    //             console.log(data)
    //         })

    // });
})