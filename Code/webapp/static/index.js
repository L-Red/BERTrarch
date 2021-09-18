// import axios from 'axios';
const vm = new Vue({ // Again, vm is our Vue instance's name for consistency.
  el: '#vm',
  delimiters: ['[[', ']]'],
  data: {
      greeting: 'Hello, Bitchaesss!',
      upload: true,
      inputs : [
        {sentence: 'abc'}
      ],
      outputs: [
        {'sentence': ['q', 'r', 'd']},
        {'sentence': ['f', 'f', 'f']}
      ],
      payload: {
        inputs: []
      },
      labels: [0, 1, 2],
      showTable: true,
      showLoader: false,
  },
  methods: {
    add () {
      this.inputs.push({sentence: ''})
    },
    doUpload () {
      this.upload = true
    },
    dontUpload () {
      this.upload = false
    },
    getSentences () {
      // this.description = this.form.text
      const path = '/sentences'
      this.showTable = false
      this.showLoader = true
      this.payload.inputs = this.inputs
      console.log('starting transfer')
      axios.post(path, this.payload)
        .then((res) => {
          console.log('ending transfer')
          console.log(res.data.outputs);
          this.outputs = res.data.outputs;
          this.labels = res.data.labels;
          this.showTable = true
          this.showLoader = false
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
          this.description = "error";
          this.show = true;
        })
    }
  }
})