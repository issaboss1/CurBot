function redirectToStreamlit() {
            
  window.location.href = "https://curbotchat.vercel.app/";
}

document.addEventListener("DOMContentLoaded", function () {           
    const contactButton = document.getElementById("contact-button");           
    const topicsButton = document.getElementById("topics-button");    
    const aboutButton = document.getElementById("about-button");           
    const faqButton = document.getElementById("faq-button");           
    const resourcesButton = document.getElementById("resource-container");
    const menuButton = document.getElementById("menu-button");
    const menuOverlay = document.getElementById("menu-overlay");
    const closeMenuLink = document.getElementById("close-menu");
    const chatBotButton =  document.getElementById("chatbot-container");
    const getStartedButton = document.getElementById("get-started-button");
    const letsButton = document.getElementById("lets-button");

    function fadeAndRedirect(destination) {
        // Fade out the container
        document.body.style.opacity = 0;
        // Wait for the transition to complete
        setTimeout(function () {
          // Redirect to the destination
          window.location.href = destination;
        }, 0); // 300ms is the duration of the transition
    } 

    const buttons = document.querySelectorAll('.common-hover');

    buttons.forEach(button => {
      button.addEventListener('mouseover', function () {
        buttons.forEach(btn => btn.classList.add('hovered'));
      });

      button.addEventListener('mouseout', function () {
        buttons.forEach(btn => btn.classList.remove('hovered'));
      });
    });

    // Event listener for the "Menu" button click
    menuButton.addEventListener("click", function () {
        // Toggle the active class to show/hide the overlay
        menuOverlay.classList.toggle("active");
    });   
    
    closeMenuLink.addEventListener("click", function (event) {
        event.preventDefault(); // Prevent the default link behavior (navigating)
        // Close the menu by removing the active class
        menuOverlay.classList.remove("active");
        });

        
 
    contactButton.addEventListener("click", function () {
        fadeAndRedirect("contact.html");
    });          

    topicsButton.addEventListener("click", function () {
        window.location.href = "topics.html";
    });

    aboutButton.addEventListener("click", function () {
        window.location.href = "about.html";
    });          

    faqButton.addEventListener("click", function () {
        window.location.href = "FAQ.html";
    });

    resourcesButton.addEventListener("click", function () {
        fadeAndRedirect("Resources.html");
    });

    chatBotButton.addEventListener("click", function () {
        window.location.href = "https://curbotchat.vercel.app/";
      });

    getStartedButton.addEventListener("click", function () {
      window.location.href = "https://curbotchat.vercel.app/";
    });

  letsButton.addEventListener("click", function () {
      fadeAndRedirect("terms_conditions_box.html");
  });
  
       
});