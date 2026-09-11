class PCMPlayerProcessor extends AudioWorkletProcessor {

    constructor() {

        super();

        this.queue = [];

        this.currentChunk = null;

        this.currentIndex = 0;


        this.port.onmessage = (event) => {

            if (event.data.type === "audio") {

                const int16 =
                    new Int16Array(
                        event.data.buffer
                    );

                this.queue.push(int16);
            }


            if (event.data.type === "clear") {

                this.queue = [];

                this.currentChunk = null;

                this.currentIndex = 0;
            }

        };

    }


    process(inputs, outputs) {

        const output =
            outputs[0];

        const channel =
            output[0];


        for (
            let i = 0;
            i < channel.length;
            i++
        ) {

            if (
                !this.currentChunk ||
                this.currentIndex >=
                this.currentChunk.length
            ) {

                if (
                    this.queue.length === 0
                ) {

                    channel[i] = 0;

                    continue;
                }


                this.currentChunk =
                    this.queue.shift();

                this.currentIndex = 0;

            }


            channel[i] =
                this.currentChunk[
                    this.currentIndex++
                ] / 32768;

        }


        return true;

    }

}


registerProcessor(
    "pcm-player",
    PCMPlayerProcessor
);