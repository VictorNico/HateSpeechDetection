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
                    <img src="@/assets/lotus.webp"
                         style="width: 185px;" alt="logo">
                    <h4 class="mt-1 mb-5 pb-1">J.U.IN Chat APP <br/> 2024 </h4>
                  </div>

                  <form>
                    <p>Please connect to your discussion channel</p>
                    <select class="form-select" aria-label="Default select example"  v-model="selectedRoom">
                      <option selected value="">Open this select menu</option>
                      <option :key="option._id" :value="option.name" v-for="option in roomList">Room ID : {option.name}</option>
                    </select>
                    <div class="my-2">
                      <input type="text" class="form-control" id="RoomId" placeholder="Room name ID" v-model="roomId">
                    </div>
                    <div class="d-grid gap-2 my-2" v-if="canJoin">
                        <button class="btn btn-warning text-white fw-bold fs-5" type="button" >Join Room</button>
                    </div>
                    <div class="d-grid gap-2 my-2">
                      <button class="btn btn-warning text-white fw-bold fs-5" type="button">Create Room</button>
                    </div>

                  </form>

                </div>
              </div>
              <div class="col-lg-6 d-flex align-items-center gradient-custom-2">
                <div class="text-white px-3 py-4 p-md-5 mx-md-4 fs-2">
                  <h4 class="mb-4 fw-bold ">"AI in Social Security"</h4>
                  <p class="mb-0 fw-semibold">We propose an AI solution to help you detect hate speech and offensive language</p>
                </div>
              </div>
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
      roomId: "",
      selectedRoom: "",
      roomList: [],
      usedName: false,
    }
  },
  computed:{
    canJoin(){return ((this.roomId.localeCompare("") !== 0) || (this.selectedRoom.localeCompare("") !== 0) && (this.usedName === false) )},

  },
  methods:{
    joinRoom(){
      alert("hh")
    },
    createRoom(){
      return 0
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