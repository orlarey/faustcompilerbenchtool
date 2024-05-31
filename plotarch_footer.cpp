

int main(int argc, char* argv[])
{
    mydsp* d = new mydsp();

    if (d == nullptr) {
        std::cerr << "Failed to create DSP object\n";
        return 1;
    }
    d->init(44100);

    // Create the input buffers
    FAUSTFLOAT* inputs[256];
    for (int i = 0; i < d->getNumInputs(); i++) {
        inputs[i] = new FAUSTFLOAT[NBSAMPLES];
        for (int j = 0; j < NBSAMPLES; j++) {
            inputs[i][j] = 0.0;
        }
        inputs[i][0] = 1.0;
    }

    // Create the output buffers
    FAUSTFLOAT* outputs[256];
    for (int i = 0; i < d->getNumOutputs(); i++) {
        outputs[i] = new FAUSTFLOAT[NBSAMPLES];
        for (int j = 0; j < NBSAMPLES; j++) {
            outputs[i][j] = 0.0;
        }
    }

    // Compile the Impulse Response
    d->compute(NBSAMPLES, inputs, outputs);

    // Print the NBSAMPLES of the impulse response
    for (int i = 0; i < NBSAMPLES; i++) {
        std::cout << i << ": ";
        for (int j = 0; j < d->getNumOutputs(); j++) {
            std::cout << outputs[j][i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
