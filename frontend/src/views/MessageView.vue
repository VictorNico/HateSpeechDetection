<template>
  <div class="d-flex">
      <div class="col-auto col-md-3 col-xl-2 px-sm-2 px-0 bg-light min-vh-100" v-if="showSidebar">
        <div class="sticky-top bg-light">
          <a href="/" class="d-flex align-items-center justify-content-center pb-3 mb-md-0 me-md-auto text-white text-decoration-none  border-1">
            <img src="@/assets/lotus.webp" alt="logo" width="150" height="150" class="rounded-circle">
          </a>
          <span class="fs-5  d-inline  text-dark fw-semibold">J.U.IN Chat App<br/>2024</span>
          <hr>
        </div>
        <div class="list-group overflow-auto">
          <small>And some small print.</small>
        </div>
      </div>
      <div class="col min-vh-100 main-content position-relative min-vh-100 d-flex flex-column h-100 border-radius-lg">
        <span class="d-inline-flex align-items-center p-1 sticky-top bg-white">
          <button type="button" :class="['btn bi' ,!showSidebar?'bi-justify':'bi-x-lg']" @click="()=>showSidebar = !showSidebar"></button>
        </span>

        <div class="list-group overflow-auto">
          <div class="chat">
            <div class="chat-history">
              <ul class="m-b-0">
<!--                <li class="clearfix">-->
<!--                  <div class="message-data text-right">-->
<!--                    <span class="message-data-time">10:10 AM, Today</span>-->
<!--                    <img src="https://bootdey.com/img/Content/avatar/avatar7.png" alt="avatar">-->
<!--                  </div>-->
<!--                  <div class="message other-message float-right"> Hi Aiden, how are you? How is the project coming along? </div>-->
<!--                </li>-->
                <li :class="['clearfix']" :key="msg.messageText" v-for="msg in ListOfMessages">
                  <div class="message-data">
                    <span :class="['message-data-time',targetColor(msg.prediction)]">{{  target(msg.prediction) }}</span>
                  </div>
                  <div class="message my-message">{{ msg.messageText }}</div>
                </li>
<!--                <li class="clearfix">-->
<!--                  <div class="message-data">-->
<!--                    <span class="message-data-time">10:15 AM, Today</span>-->
<!--                  </div>-->
<!--                  <div class="message my-message">Project has been already finished and I have results to show you.</div>-->
<!--                </li>-->
              </ul>
            </div>
          </div>

        </div>
        <footer class="py-1 footer mt-auto bg-light  sticky-bottom">
          <div class="input-group mb-1">
            <input type="text" class="form-control" aria-label="Message" placeholder="Send a message" v-model="messageText" @keypress.enter="send">
            <button :class="['input-group-text w-5',!canSend === true?'bg-secondary':'bg-warning']" @click="send" :disabled="!canSend"><i class="bi bi-send-fill text-white "></i></button>
          </div>
        </footer>
      </div>
    </div>
</template>
<script>
// @ is an alias to /src
// import bg from "@/assets/WhatsApp Image 2024-05-13 at 20.19.26.jpeg";

import axios from "axios"


export default {
  name: 'MessageView',
  components: {
  },
  data(){
    return {
      RoomId: undefined,
      showSidebar: false,
      ListOfMessages: [],
      messageText: ""

    }
  },
  mounted(){
    this.RoomId = this.$route.params.id;
  },
  computed:{
    canSend(){return ((this.messageText.localeCompare("") !== 0))},

  },
  methods:{
    target(input){
      let flag = ''
      if(input.localeCompare("2") == 0){
        flag = "normal message"
      }
      else if(input.localeCompare("1") == 0){
        flag = "offensive language"
      }
      else if(input.localeCompare("0") == 0){
        flag = "hate speech"
      }
      return flag
    },
    targetColor(input){
      let flag = ''
      if(input.localeCompare("2") == 0){
        flag = "text-primary"
      }
      else if(input.localeCompare("1") == 0){
        flag = "text-warning"
      }
      else if(input.localeCompare("0") == 0){
        flag = "text-danger"
      }
      return flag
    },
    send(){
      let data = {
        messageText: [`@ksjfdkfd ${this.messageText}`]
      }
      const config = {
        method: 'post',
        url: `${process.env.VUE_APP_API_URL}/api/predict`,
        headers: {
          'Content-Type': 'application/json',
        },
        data : data
      };
      axios(config)
          .then( (response) => {
            console.log(JSON.stringify(response.data));
            this.ListOfMessages.push({messageText: this.messageText, prediction: response.data.prediction});
            this.messageText = "";
          })
    },
  }
}
</script>
<style>
.bg-juin {
  background: linear-gradient(to right, #e4883b, #da7d3a, #bf794f, #916949) !important;
}

.card {
  background: #fff;
  transition: .5s;
  border: 0;
  margin-bottom: 30px;
  border-radius: .55rem;
  position: relative;
  width: 100%;
  box-shadow: 0 1px 2px 0 rgb(0 0 0 / 10%);
}
.chat-app .people-list {
  width: 280px;
  position: absolute;
  left: 0;
  top: 0;
  padding: 20px;
  z-index: 7
}

.chat-app .chat {
  margin-left: 280px;
  border-left: 1px solid #eaeaea
}

.people-list {
  -moz-transition: .5s;
  -o-transition: .5s;
  -webkit-transition: .5s;
  transition: .5s
}

.people-list .chat-list li {
  padding: 10px 15px;
  list-style: none;
  border-radius: 3px
}

.people-list .chat-list li:hover {
  background: #efefef;
  cursor: pointer
}

.people-list .chat-list li.active {
  background: #efefef
}

.people-list .chat-list li .name {
  font-size: 15px
}

.people-list .chat-list img {
  width: 45px;
  border-radius: 50%
}

.people-list img {
  float: left;
  border-radius: 50%
}

.people-list .about {
  float: left;
  padding-left: 8px
}

.people-list .status {
  color: #999;
  font-size: 13px
}

.chat .chat-header {
  padding: 15px 20px;
  border-bottom: 2px solid #f4f7f6
}

.chat .chat-header img {
  float: left;
  border-radius: 40px;
  width: 40px
}

.chat .chat-header .chat-about {
  float: left;
  padding-left: 10px
}

.chat .chat-history {
  padding: 20px;
}

.chat .chat-history ul {
  padding: 0
}

.chat .chat-history ul li {
  list-style: none;
  margin-bottom: 30px
}

.chat .chat-history ul li:last-child {
  margin-bottom: 0px
}

.chat .chat-history .message-data {
  margin-bottom: 15px
}

.chat .chat-history .message-data img {
  border-radius: 40px;
  width: 40px
}

.chat .chat-history .message-data-time {
  color: #434651;
  padding-left: 6px
}

.chat .chat-history .message {
  color: #444;
  padding: 18px 20px;
  line-height: 26px;
  font-size: 16px;
  border-radius: 7px;
  display: inline-block;
  position: relative
}

.chat .chat-history .message:after {
  bottom: 100%;
  left: 7%;
  border: solid transparent;
  content: " ";
  height: 0;
  width: 0;
  position: absolute;
  pointer-events: none;
  border-bottom-color: #fff;
  border-width: 10px;
  margin-left: -10px
}

.chat .chat-history .my-message {
  background: #efefef
}

.chat .chat-history .my-message:after {
  bottom: 100%;
  left: 30px;
  border: solid transparent;
  content: " ";
  height: 0;
  width: 0;
  position: absolute;
  pointer-events: none;
  border-bottom-color: #efefef;
  border-width: 10px;
  margin-left: -10px
}

.chat .chat-history .other-message {
  background: #e8f1f3;
  text-align: right
}

.chat .chat-history .other-message:after {
  border-bottom-color: #e8f1f3;
  left: 93%
}

.chat .chat-message {
  padding: 20px
}

.online,
.offline,
.me {
  margin-right: 2px;
  font-size: 8px;
  vertical-align: middle
}

.online {
  color: #86c541
}

.offline {
  color: #e47297
}

.me {
  color: #1d8ecd
}

.float-right {
  float: right
}

.clearfix:after {
  visibility: hidden;
  display: block;
  font-size: 0;
  content: " ";
  clear: both;
  height: 0
}

@media only screen and (max-width: 767px) {
  .chat-app .people-list {
    height: 465px;
    width: 100%;
    overflow-x: auto;
    background: #fff;
    left: -400px;
    display: none
  }
  .chat-app .people-list.open {
    left: 0
  }
  .chat-app .chat {
    margin: 0
  }
  .chat-app .chat .chat-header {
    border-radius: 0.55rem 0.55rem 0 0
  }
  .chat-app .chat-history {
    height: 300px;
    overflow-x: auto
  }
}

@media only screen and (min-width: 768px) and (max-width: 992px) {
  .chat-app .chat-list {
    height: 650px;
    overflow-x: auto
  }
  .chat-app .chat-history {
    height: 600px;
    overflow-x: auto
  }
}

@media only screen and (min-device-width: 768px) and (max-device-width: 1024px) and (orientation: landscape) and (-webkit-min-device-pixel-ratio: 1) {
  .chat-app .chat-list {
    height: 480px;
    overflow-x: auto
  }
  .chat-app .chat-history {
    height: calc(100vh - 350px);
    overflow-x: auto
  }
}
</style>