// Sentinel-X - manejo de las pestañas del Threat Scanner (URL / Email)
document.addEventListener("DOMContentLoaded", function () {
  var tabs = document.querySelectorAll(".tab[data-tab]");
  var tipoInput = document.getElementById("tipo-input");
  var urlRow = document.getElementById("url-row");
  var emailRow = document.getElementById("email-row");
  var emailSubmit = document.getElementById("email-submit");
  var scanHint = document.getElementById("scan-hint");
  var inputUrl = document.getElementById("valor-url");
  var inputEmail = document.getElementById("valor-email");
  var form = document.getElementById("scan-form");

  function activar(tab) {
    tabs.forEach(function (t) {
      t.classList.toggle("active", t.dataset.tab === tab);
    });
    if (tab === "email") {
      urlRow.style.display = "none";
      emailRow.style.display = "flex";
      emailSubmit.style.display = "block";
      scanHint.style.display = "block";
      tipoInput.value = "email";
      inputUrl.name = "";
      inputEmail.name = "valor";
    } else {
      urlRow.style.display = "flex";
      emailRow.style.display = "none";
      emailSubmit.style.display = "none";
      scanHint.style.display = "none";
      tipoInput.value = "url";
      inputUrl.name = "valor";
      inputEmail.name = "";
    }
  }

  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      activar(t.dataset.tab);
    });
  });

  if (form) {
    form.addEventListener("submit", function (e) {
      var activo = tipoInput.value === "email" ? inputEmail : inputUrl;
      if (!activo.value.trim()) {
        e.preventDefault();
        activo.focus();
      }
    });
  }
});
