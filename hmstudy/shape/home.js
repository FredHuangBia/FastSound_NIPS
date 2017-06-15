var imgArray;
var page_count = 0;
var num_page = 10;
var label = new Array(12);
var played = new Array(7);
var chosen = new Array(6);
var theme;

var images = new Array()


function changePage(){

  // 
  num_page = imgArray.length/2;     

  document.getElementById('bt1').disabled = true;
  document.getElementById('bt2').disabled = true;
  document.getElementById('bt3').disabled = true;
  document.getElementById('bt4').disabled = true;
  document.getElementById('bt5').disabled = true;
  document.getElementById('bt6').disabled = true;

  document.getElementById('choice0').style.visibility="hidden";
  document.getElementById('choice1').style.visibility="hidden";
  document.getElementById('choice2').style.visibility="hidden";
  document.getElementById('choice3').style.visibility="hidden";
  document.getElementById('choice4').style.visibility="hidden";
  document.getElementById('choice5').style.visibility="hidden";

  // document.getElementById('label').innerHTML = imgArray[page_count * 3];

  document.getElementById('s0a').setAttribute("src", imgArray[page_count*2]);
  document.getElementById('s0b').setAttribute("src", imgArray[page_count*2+1]);
  document.getElementById('a0').load();

  played[0] = false
  chosen[0] = false
  chosen[1] = false
  chosen[2] = false
  chosen[3] = false
  chosen[4] = false
  chosen[5] = false

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

        

        for (i = 1; i <= 6; i++) {
          document.getElementById('img'+i).setAttribute("src", 'data/loading.gif');
        }
        document.getElementById('img1').setAttribute("src", 'data/shape/1.png');
        document.getElementById('img2').setAttribute("src", 'data/shape/3.png');
        document.getElementById('img3').setAttribute("src", 'data/shape/6.png');
        document.getElementById('img4').setAttribute("src", 'data/shape/10.png');
        document.getElementById('img5').setAttribute("src", 'data/shape/13.png');
        document.getElementById('img6').setAttribute("src", 'data/shape/14.png');
        
        document.getElementById('s1a').setAttribute("src", 'data/shape/1.m4a');
        document.getElementById('s1b').setAttribute("src", 'data/shape/1.ogg');
        document.getElementById('a1').load();      
        document.getElementById('s2a').setAttribute("src", 'data/shape/3.m4a');
        document.getElementById('s2b').setAttribute("src", 'data/shape/3.ogg');
        document.getElementById('a2').load();     
        document.getElementById('s3a').setAttribute("src", 'data/shape/6.m4a');
        document.getElementById('s3b').setAttribute("src", 'data/shape/6.ogg');
        document.getElementById('a3').load();     
        document.getElementById('s4a').setAttribute("src", 'data/shape/10.m4a');
        document.getElementById('s4b').setAttribute("src", 'data/shape/10.ogg');
        document.getElementById('a4').load();     
        document.getElementById('s5a').setAttribute("src", 'data/shape/13.m4a');
        document.getElementById('s5b').setAttribute("src", 'data/shape/13.ogg');
        document.getElementById('a5').load();  
        document.getElementById('s6a').setAttribute("src", 'data/shape/14.m4a');
        document.getElementById('s6b').setAttribute("src", 'data/shape/14.ogg');
        document.getElementById('a6').load();  

        chosen[0]=true
        chosen[1]=true
        chosen[2]=true
        chosen[3]=true
        chosen[4]=true
        chosen[5]=true        
        played[0]=true
        played[1]=false
        played[2]=false
        played[3]=false
        played[4]=false
        played[5]=false
        played[6]=false


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
  
  // document.getElementById('choice').innerHTML = "I think object "+id+" matches the best.";
  document.getElementById('choice' + id).style.visibility="visible";
  chosen[id] = true
  if (id%2 == 0){
    chosen[id+1] = false
    document.getElementById('choice' + (id+1) ).style.visibility="hidden";
  } else {
    chosen[id-1] = false
    document.getElementById('choice' + (id-1) ).style.visibility="hidden";
  }

  var results = [0, 0, 0];
  if (chosen[0]) {
    results[0] = 1
  }
  if (chosen[2]) {
    results[1] = 1
  }
  if (chosen[4]) {
    results[2] = 1
  }

  label[page_count] = results[0]*1 + results[1]*2 + results[2]*4;


  if ( (chosen[0] || chosen[1]) && (chosen[2] || chosen[3]) && (chosen[4] || chosen[5]) ){
    if (page_count < num_page - 1) {		
      document.getElementById('nextButton').disabled = false;
    } else {
      document.getElementById('submitButton').disabled = false;
    }
  }
};


function audioPlayed(id){
  played[id] = true;
  if (played[0] && played[1] && played[2] && played[3] && played[4] && played[5] && played[6]) {
    document.getElementById('bt1').disabled = false;
    document.getElementById('bt2').disabled = false;
    document.getElementById('bt3').disabled = false;
    document.getElementById('bt4').disabled = false;
    document.getElementById('bt5').disabled = false;
    document.getElementById('bt6').disabled = false;
    // document.getElementById('bt5').disabled = false;
    if( (chosen[0] || chosen[1]) && (chosen[2] || chosen[3]) && (chosen[4] || chosen[5]) ){
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
  var child3 = document.getElementById("v5");
  child3.parentNode.removeChild(child3);
  var child4 = document.getElementById("v6");
  child4.parentNode.removeChild(child4);
  document.getElementById('a0').style.visibility="visible";
  document.getElementById('instruction').innerHTML="Instructions";
  document.getElementById('subins1').innerHTML="You will be given a short audio clip of an object falling onto the ground. ";
  document.getElementById('subins2').innerHTML="Please listen to the sound clip and answer three questions.";
  document.getElementById('quest').innerHTML="Listen to the sound clip and answer three questions.";
  document.getElementById('requirement').innerHTML="Please answer the following questions <strong>according to the given sound clip</strong>:";
  document.getElementById('bt1').style.visibility="visible";
  document.getElementById('bt2').style.visibility="visible";
  document.getElementById('bt3').style.visibility="visible";
  document.getElementById('bt4').style.visibility="visible";
  document.getElementById('bt5').style.visibility="visible";
  document.getElementById('bt6').style.visibility="visible";
  document.getElementById('nextButton').setAttribute("value", 'Next');
  document.getElementById('nextButton').setAttribute("onclick", 'clickNext()'); 

  document.getElementById('a1').style.visibility="visible";
  document.getElementById('a2').style.visibility="visible";
  document.getElementById('a3').style.visibility="visible";
  document.getElementById('a4').style.visibility="visible";
  document.getElementById('a5').style.visibility="visible";
  document.getElementById('a6').style.visibility="visible";

  document.getElementById('choice0').style.fontSize="20px";
  document.getElementById('choice1').style.fontSize="20px";
  document.getElementById('choice2').style.fontSize="20px";
  document.getElementById('choice3').style.fontSize="20px";
  document.getElementById('choice4').style.fontSize="20px";
  document.getElementById('choice5').style.fontSize="20px";

  chosen[0] = false
  chosen[1] = false
  chosen[2] = false
  chosen[3] = false
  chosen[4] = false
  chosen[5] = false
  played[0]=false
  played[1]=true
  played[2]=true
  played[3]=true
  played[4]=true
  played[5]=true
  played[6]=true
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
