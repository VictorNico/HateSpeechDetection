<template>
  <section class="h-100 gradient-form text-3xl" style="background-color: #eee;">
    <div class="container py-2 h-100">
      <div class="row d-flex justify-content-center align-items-center h-100">
        <div class="col-xl-12">
          <div class="card rounded-3 text-black">
            <div class="row g-0">
              <div class="col-lg-6">
                <div class="card-body p-md-5 mx-md-4">

                  <div class="text-center">
                    <img src="https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-login-form/lotus.webp"
                         style="width: 185px;" alt="logo">
                    <h4 class="mt-1 mb-5 pb-1">J.U.IN Chat APP</h4>
                  </div>

                  <form>
                    <p>Please connect to your discussion channel</p>
                    <select class="form-select" aria-label="Default select example">
                      <option selected>Open this select menu</option>
                      <option value="1">One</option>
                      <option value="2">Two</option>
                      <option value="3">Three</option>
                    </select>
                    <div class="d-grid gap-2 my-2">
                        <button class="btn btn-warning text-white fw-bold fs-5" type="button">Join</button>
                    </div>

                  </form>

                </div>
              </div>
              <div class="col-lg-6 d-flex align-items-center gradient-custom-2" :style="img">
                <div class="text-white px-3 py-4 p-md-5 mx-md-4 fs-2">
                  <h4 class="mb-4 fw-bold ">"AI in Social Security Mitigation"</h4>
                  <p class="mb-0 fw-semibold">We propose an AI solution to help you mitigate hate speech and offensive language</p>
                </div>
              </div>
<!--              <div-->
<!--                  class="top-0 my-auto text-center col-6 d-lg-flex d-none h-100 pe-0 position-absolute end-0 justify-content-center flex-column"-->
<!--              >-->
<!--                <div-->
<!--                    class="position-relative bg-gradient-primary h-100 m-3 px-7 border-radius-lg d-flex flex-column justify-content-center overflow-hidden"-->
<!--                    :style="img-->
<!--                "-->
<!--                >-->
<!--                  <span class="mask bg-gradient-success opacity-6"></span>-->
<!--                  <h4-->
<!--                      class="mt-5 text-white font-weight-bolder position-relative"-->
<!--                  >-->
<!--                    "Reduce school drop out."-->
<!--                  </h4>-->
<!--                  <p class="text-white position-relative">-->
<!--                    We offer you, simple, easy used and powerful app to predict the drop out probabilities of your students.-->
<!--                  </p>-->
<!--                </div>-->
<!--              </div>-->
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
// @ is an alias to /src
// import bg from "@/assets/WhatsApp Image 2024-05-13 at 20.19.26.jpeg";
export default {
  name: 'HomeView',
  components: {
  },
  data(){
    return {
      // img: [
      //   `background-image: url(${bg});`,
      //   'background-size: cover;'
      // ]
    }
  }
}
</script>
<style>
.gradient-custom-2 {
  /* fallback for old browsers */
  background: #E56B06;
  /* Chrome 10-25, Safari 5.1-6 */
  background: -webkit-linear-gradient(to right, #E56B06, #FFC08B, #dd3675, #FFC08B);
  /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+*/
  background: linear-gradient(to right, #E56B06, #d87a36, #d58251, #e0904e);


}

.btn-warning {
  background: linear-gradient(to right, #E56B06, #d87a36, #d58251, #e0904e) !important;
}
.desc {
  background-image: url("@/assets/WhatsApp Image 2024-05-13 at 20.19.26.jpeg");
  background-size: cover;
}
@media (min-width: 768px) {
  .gradient-form {
    height: 100vh !important;
  }
}
@media (min-width: 769px) {
  .gradient-custom-2 {
    border-top-right-radius: .3rem;
    border-bottom-right-radius: .3rem;
  }
}
.desc {
  text-align: center;
  text-autospace: ideograph-space;

}
</style>