var imgArray;
var page_count = 0;
var num_page = 10;
var label = new Array(12);
var played = new Array(9);
var chosen = false
var theme;

var images = new Array()
var kvArray = new Array(3)
kvArray[0]='lower'
kvArray[1]='higher'


function changePage(){

  // 
  num_page = imgArray.length/2;     

  document.getElementById('bt1').disabled = true;
  document.getElementById('bt2').disabled = true;
  // document.getElementById('bt3').disabled = true;
  // document.getElementById('bt4').disabled = true;
  // document.getElementById('bt5').disabled = true;

  // document.getElementById('label').innerHTML = imgArray[page_count * 3];

  document.getElementById('s0a').setAttribute("src", imgArray[page_count*2]);
  document.getElementById('s0b').setAttribute("src", imgArray[page_count*2+1]);
  document.getElementById('a0').load();

  played[0] = false;
  chosen = false

  document.getElementById('choice').innerHTML = "Please listen to the sound before making your choice.";
  document.getElementById('choice').style.visibility = "hidden"

  document.getElementById('nextButton').disabled = true;
  document.getElementById('submitButton').disabled = true;

  document.getElementById("pageId").innerHTML = "Page " + (page_count + 1) + " of " + num_page;
}
  

  $(document).ready(function(){	
    var filename = gup('filename');
    filename = "fileList/" + filename ;
    
    var xmlhttp;
    if (window.XMLHttpRequest) {
      // code for IE7+, Firefox, Chrome, Opera, Safari
      xmlhttp = new XMLHttpRequest();
    } else {
      // code for IE6, IE5
      xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }	

    xmlhttp.open("GET", filename, true);
    xmlhttp.send();
    xmlhttp.onreadystatechange = function() {
      if (xmlhttp.readyState == 4 && xmlhttp.status == 200){			
        imgArray = xmlhttp.responseText.split("\n");

        for (i = 1; i <= 2; i++) {
          document.getElementById('img'+i).setAttribute("src", 'data/loading.gif');
        }
        document.getElementById('img1').setAttribute("src", 'data/shape/1.png');
        document.getElementById('img2').setAttribute("src", 'data/shape/2.png');
        document.getElementById('img3').setAttribute("src", 'data/shape/3.png');
        document.getElementById('img4').setAttribute("src", 'data/shape/4.png');
        // document.getElementById('img3').setAttribute("src", 'data/shape/3.png');
        // document.getElementById('img4').setAttribute("src", 'data/shape/4.png');
        // document.getElementById('img5').setAttribute("src", 'data/shape/5.png');
        
        document.getElementById('s1a').setAttribute("src", 'data/shape/1.m4a');
        document.getElementById('s1b').setAttribute("src", 'data/shape/1.ogg');
        document.getElementById('a1').load();      
        document.getElementById('s2a').setAttribute("src", 'data/shape/2.m4a');
        document.getElementById('s2b').setAttribute("src", 'data/shape/2.ogg');
        document.getElementById('a2').load();     
        document.getElementById('s3a').setAttribute("src", 'data/shape/3.m4a');
        document.getElementById('s3b').setAttribute("src", 'data/shape/3.ogg');
        document.getElementById('a3').load();      
        document.getElementById('s4a').setAttribute("src", 'data/shape/4.m4a');
        document.getElementById('s4b').setAttribute("src", 'data/shape/4.ogg');
        document.getElementById('a4').load();     
        // document.getElementById('s3a').setAttribute("src", 'data/shape/3.wav');
        // document.getElementById('s3b').setAttribute("src", 'data/shape/3.wav');
        // document.getElementById('a3').load();     
        // document.getElementById('s4a').setAttribute("src", 'data/shape/4.wav');
        // document.getElementById('s4b').setAttribute("src", 'data/shape/4.wav');
        // document.getElementById('a4').load();     
        // document.getElementById('s5a').setAttribute("src", 'data/shape/5.wav');
        // // document.getElementById('s5b').setAttribute("src", 'data/shape/5.wav');
        // document.getElementById('a5').load();  

        chosen=true
        played[0]=true
        played[1]=false
        played[2]=false
        played[3]=false
        played[4]=false
        played[5]=false
        played[6]=false
        played[7]=false
        played[8]=false

      }
    }	

    document.getElementById("assignmentId").value = gup('assignmentId');
    //
    // Check if the worker is PREVIEWING the HIT or if they've ACCEPTED the HIT
    //
    if (gup('assignmentId') == "ASSIGNMENT_ID_NOT_AVAILABLE") {
      // If we're previewing, disable the button and give it a helpful message
      document.getElementById('submitButton').disabled = true;
      document.getElementById('nextButton').disabled = true;

      document.getElementById('submitButton').value = "You must ACCEPT the HIT before you can submit the results.";
      document.getElementById('submitButton').css("width", "350px");
    } else {
      var form = document.getElementById('mturk_form');
      if (document.referrer && ( document.referrer.indexOf('workersandbox') != -1) ) {
        form.action = "https://workersandbox.mturk.com/mturk/externalSubmit";
      } else {
        form.action = "https://www.mturk.com/mturk/externalSubmit";
      }
    }
  });

function madeChoice(id){	
  label[page_count] = id;
  document.getElementById('choice').innerHTML = "I think the "+kvArray[id-1]+" one matches better.";
  chosen = true
  document.getElementById('choice').style.visibility = "visible";

  if (chosen){
    if (page_count < num_page - 1) {		
      document.getElementById('nextButton').disabled = false;
    } else {
      document.getElementById('submitButton').disabled = false;
    }
  }
};


function audioPlayed(id){
  played[id] = true;
  if (played[0] && played[5] && played[6] && played[7] && played[8]) {
    document.getElementById('bt1').disabled = false;
    document.getElementById('bt2').disabled = false;
    // document.getElementById('bt3').disabled = false;
    // document.getElementById('bt4').disabled = false;
    // document.getElementById('bt5').disabled = false;
    if(chosen){
      if (page_count < num_page - 1) {    
        document.getElementById('nextButton').disabled = false;
      } else {
        document.getElementById('submitButton').disabled = false;
      }      
    }
  }
};


function clickNext(){	
  ++page_count;	
  changePage()
};


function start(){  
  var child1 = document.getElementById("v1");
  child1.parentNode.removeChild(child1);
  var child2 = document.getElementById("v2");
  child2.parentNode.removeChild(child2);
  var child3 = document.getElementById("v3");
  child3.parentNode.removeChild(child3);
  var child4 = document.getElementById("v4");
  child4.parentNode.removeChild(child4);
  // var child3 = document.getElementById("v3");
  // child3.parentNode.removeChild(child3);
  // var child4 = document.getElementById("v4");
  // child4.parentNode.removeChild(child4);
  document.getElementById('a0').style.visibility="visible";
  document.getElementById('instruction').innerHTML="Instructions";
  document.getElementById('subins1').innerHTML="You will listen to a sound clip and see 2 pairs of images corresponding to 2 different initial heights.";
  document.getElementById('subins2').innerHTML="Imagine an object falling onto the ground <strong>from a specific height</strong>. Please choose the height that <strong>better</strong> matches the sound clip.";
  document.getElementById('quest').innerHTML="Which initial height better matches the sound?";
  document.getElementById('requirement').innerHTML="Listen to the sound clip and select.";
  document.getElementById('bt1').style.visibility="visible";
  document.getElementById('bt2').style.visibility="visible";
  document.getElementById('a1').style.visibility="visible";
  document.getElementById('a2').style.visibility="visible";
  document.getElementById('a3').style.visibility="visible";
  document.getElementById('a4').style.visibility="visible";
  document.getElementById('img1').style.visibility="visible";
  document.getElementById('img2').style.visibility="visible";
  document.getElementById('img3').style.visibility="visible";
  document.getElementById('img4').style.visibility="visible";
  // document.getElementById('bt3').style.visibility="visible";
  // document.getElementById('bt4').style.visibility="visible";
  document.getElementById('nextButton').setAttribute("value", 'Next');
  document.getElementById('nextButton').setAttribute("onclick", 'clickNext()'); 
  chosen = false
  played[0]=false
  played[1]=true
  played[2]=true
  played[3]=true
  played[4]=true
  played[5]=true
  played[6]=true
  played[7]=true
  played[8]=true
  changePage(); 
};


function clickButton(){		
  var action = $("#mturk_form").attr("action");
  action += "?";
  action += "assignmentId=";
  action += $("#assignmentId").attr("value");
  for (var i = 1; i <= num_page; ++i) {
    action += "&";
    action += i;
    action += "=";
    action += label[i - 1];
  }
  $("#mturk_form").attr("action", action);
};



//
// This method Gets URL Parameters (GUP)
//
function gup( name )
{
  var regexS = "[\\?&]" + name + "=([^&#]*)";
  var regex = new RegExp( regexS );
  var tmpURL = window.location.href;
  var results = regex.exec( tmpURL );
  if( results == null )
    return "";
  else
    return results[1];
};

//
// This method decodes the query parameters that were URL-encoded
//
function decode(strToDecode)
{
  var encoded = strToDecode;
  return unescape(encoded.replace(/\+/g,  " "));
};
