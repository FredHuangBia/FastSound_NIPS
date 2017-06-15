#include "AudioProducer.h"
#include <QtEndian>
#include <fstream>
#include "protobuf/sploosh.pb.h"
#include "utils/term_msg.h"
#include "utils/macros.h"
#include "transfer/FMMTransferEval.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <QFile>
#include <QDataStream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


using namespace std;

static const int SR = 44100;
//static const double TS = 3;    // synthesize sound for at most 3 seconds
//static const double TS = 1;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
class WavPcmFile : public QFile {
    public:
        WavPcmFile(const QString & name, const QAudioFormat & format, QObject *parent = 0);
        bool open();
        void close();

    private:
        void writeHeader();
        bool hasSupportedFormat();
        QAudioFormat format;
};


WavPcmFile::WavPcmFile(const QString & name, const QAudioFormat & format_, QObject *parent_)
: QFile(name, parent_), format(format_)
{
}

bool WavPcmFile::hasSupportedFormat()
{
    return (format.sampleSize() == 8
    && format.sampleType() == QAudioFormat::UnSignedInt)
    || (format.sampleSize() > 8
    && format.sampleType() == QAudioFormat::SignedInt
    && format.byteOrder() == QAudioFormat::LittleEndian);
}

bool WavPcmFile::open()
{
    if (!hasSupportedFormat()) {
        setErrorString("Wav PCM supports only 8-bit unsigned samples "
        "or 16-bit (or more) signed samples (in little endian)");
        printf("Failure1\n");
        return false;
    }
    else {
        if (!QFile::open(ReadWrite | Truncate))
        {
            printf("Failure2\n");
            return false;
        }
        writeHeader();
        return true;
    }
}

void WavPcmFile::writeHeader()
{
    QDataStream out(this);
    out.setByteOrder(QDataStream::LittleEndian);

// RIFF chunk
    out.writeRawData("RIFF", 4);
    out << quint32(0); // Placeholder for the RIFF chunk size (filled by close())
    out.writeRawData("WAVE", 4);

// Format description chunk
    out.writeRawData("fmt ", 4);
    out << quint32(16); // "fmt " chunk size (always 16 for PCM)
    out << quint16(1); // data format (1 => PCM)
    out << quint16(format.channelCount());
    out << quint32(format.sampleRate());
    out << quint32(format.sampleRate() * format.channelCount()
    * format.sampleSize() / 8 ); // bytes per second
    out << quint16(format.channelCount() * format.sampleSize() / 8); // Block align
    out << quint16(format.sampleSize()); // Significant Bits Per Sample

// Data chunk
    out.writeRawData("data", 4);
    out << quint32(0); // Placeholder for the data chunk size (filled by close())

    Q_ASSERT(pos() == 44); // Must be 44 for WAV PCM
}

void WavPcmFile::close()
{
// Fill the header size placeholders
    quint32 fileSize = size();

    QDataStream out(this);
// Set the same ByteOrder like in writeHeader()
    out.setByteOrder(QDataStream::LittleEndian);
// RIFF chunk size
    seek(4);
    out << quint32(fileSize - 8);

// data chunk size
    seek(40);
    out << quint32(fileSize - 44);

    QFile::close();
}


inline bool exists_test (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


AudioProducer::AudioProducer(const QSettings& settings, const QDir& dataDir):
        device_(NULL), normalizeScale_(-1.)
{
    ////// load data
    QString filename;
    // --- load vertex map ---
    {
        QString ff = settings.value("modal/vtx_map").toString();
        QFileInfo fInfo(ff);
        filename = fInfo.isRelative() ? dataDir.filePath(ff) : ff;
    }
    load_vertex_map(filename);

    // --- load modal shape matrix ---
    {
        QString ff = settings.value("modal/shape").toString();
        QFileInfo fInfo(ff);
        filename = fInfo.isRelative() ? dataDir.filePath(ff) : ff;
    }

    modal_ = new ModalModel(
        filename.toStdString(),
        settings.value("modal/density").toDouble(),
        settings.value("modal/alpha").toDouble(),
        settings.value("modal/beta").toDouble() );

    mForce_.resize( modal_->num_modes() );

    // --- load moments ---
    {
        QString ff = settings.value("transfer/moments").toString();
        QFileInfo fInfo(ff);
        filename = fInfo.isRelative() ? dataDir.filePath(ff) : ff;
    }
    load_moments(filename);

//////////////////////////////////////////////////////////////////////////////
    use_audio_device = settings.value("audio/use_audio_device").toBool();
    TS = settings.value("audio/TS").toDouble();
//////////////////////////////////////////////////////////////////////////////
    // --- setup audio output device ---
    if(use_audio_device){
        const QString devStr = settings.value("audio/device").toString();

        foreach(const QAudioDeviceInfo &deviceInfo,
                QAudioDeviceInfo::availableDevices(QAudio::AudioOutput))
        {
            if ( devStr == deviceInfo.deviceName() )
                device_ = new QAudioDeviceInfo( deviceInfo );
        }

        if ( !device_ )
        {
            PRINT_ERROR("Cannot find the specified audio device\n");
            SHOULD_NEVER_HAPPEN(-2);
        }
    }

    // --- allocate data buffer ---
    init();
}

AudioProducer::~AudioProducer()
{
    delete device_;
    if(use_audio_device)
      delete audioOutput_;
/////////////////////////////////////////////////////////////////////////////
    delete modal_;

    for(size_t i = 0;i < transfer_.size();++ i) delete transfer_[i];
}

void AudioProducer::load_moments(const QString& filename)
{
    sploosh::ModalMoments mms;
    ifstream fin( filename.toStdString().c_str(), ios::binary );
    if ( !mms.ParseFromIstream(&fin) )
    {
        PRINT_ERROR("Cannot real protobuf file: %s\n", filename.toStdString().c_str());
        SHOULD_NEVER_HAPPEN(-1);
    }

    const int nmms = (int)mms.moment_size();
    PRINT_MSG("%d moments are detected\n", nmms);
    if ( nmms < modal_->num_modes() )
    {
        PRINT_ERROR("Number of moments (%d) is smaller than the number of modes (%d)\n", nmms, modal_->num_modes());
        SHOULD_NEVER_HAPPEN(-2);
    } // end if

    QFileInfo checkConfig(filename);
    QString dir = checkConfig.absoluteDir().absolutePath(); //.toStdString() << endl;
    transfer_.resize( modal_->num_modes() );
    for(int mi = 0;mi < modal_->num_modes();++ mi)
    {
        transfer_[mi] = new FMMTransferEval( mms.moment(mi), dir.toStdString() );
    }
}

/*
 * Vertex map to map from surface vertex to tet vertex
 */
void AudioProducer::load_vertex_map(const QString& filename)
{
    int id1, id2;

    std::ifstream fin( filename.toStdString().c_str() );
    if ( fin.fail() )
    {
        PRINT_ERROR("Cannot read file: %s\n", filename.toStdString().c_str());
        SHOULD_NEVER_HAPPEN(2);
    }
    fin >> numFixed_ >> id1;    // # of fixed vertices in tet mesh
                                // & total number of surface vertices
    PRINT_MSG("  # of fixed vertices: %d\n", numFixed_);
    vidS2T_.resize(id1);
    for(size_t i = vidS2T_.size();i > 0;-- i)
    {
        fin >> id1 >> id2;
        if ( id2 >= vidS2T_.size() )
        {
            PRINT_ERROR("Id2 is out of range in geometry file\n");
            SHOULD_NEVER_HAPPEN(3);
        }
        vidS2T_[id2] = id1;
    }
    if ( fin.fail() )
    {
        PRINT_ERROR("Error occurred while reading file: %s\n", filename.toStdString().c_str());
        SHOULD_NEVER_HAPPEN(2);
    }
    fin.close();
}

void AudioProducer::init()
{
    // setup audio format
    format_.setSampleRate(SR);
    // format_.setChannelCount( stereo_ ? 2 : 1 );
    format_.setChannelCount(1);
    format_.setSampleSize(16);
    format_.setCodec("audio/pcm");
    format_.setByteOrder(QAudioFormat::LittleEndian);
    format_.setSampleType(QAudioFormat::SignedInt);

    // now check the format
    if (use_audio_device){
        if ( !device_->isFormatSupported(format_) )
        {
            PRINT_ERROR("The specified format cannot be supported\n");
            SHOULD_NEVER_HAPPEN(-2);
        }
    }

    // create buffer
    const qint64 len =
        (SR * TS * format_.channelCount() * format_.sampleSize() / 8);
    buffer_.resize(len);
    soundBuffer_.resize( SR * TS * format_.channelCount() );

    //create the IO device
    if(use_audio_device){
        audioIO_.close();
        audioIO_.setBuffer( &buffer_ );
        audioIO_.open( QIODevice::ReadOnly );

        audioOutput_ = new QAudioOutput( *device_, format_ );
    }
}

void AudioProducer::play(const Tuple3ui& tri, const Vector3d& dir, const Point3d& cam, float amplitude)
{
    if(use_audio_device)
      audioIO_.close();

    // ====== fill the buffer ======
    //// for now, only do the single-channel synthesis
    single_channel_synthesis(tri, dir, cam, amplitude);
    /*
    unsigned char *ptr = reinterpret_cast<unsigned char *>(buffer_.data());
    const int channelBytes = format_.sampleSize() / 8;
    for(int ii = 0;ii < SR * TS; ++ ii)
    {
        const qreal x = sin( 2. * M_PI * 200. * static_cast<double>(ii) / static_cast<double>(SR) );
        const qint16 value = static_cast<qint16>(x * 32767);    // integer value

        qToLittleEndian<qint16>(value, ptr);
        ptr += channelBytes;
    }
    */

    if(use_audio_device){
        audioIO_.open( QIODevice::ReadOnly );
        audioOutput_->reset();

        audioOutput_->start( &audioIO_ );
    }

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

    int seq = 1;
    string name="audio1.wav";
    while(exists_test(name)){
      seq++;
      string name_seq = to_string(seq);
      name = "audio"+name_seq+".wav";
    }

    QString wav_name = QString::fromStdString(name);

    WavPcmFile *m_file = new WavPcmFile(wav_name, format_);
    if(m_file->open()) {
        //audioOutput_->start(m_file);
        m_file->write(buffer_);
        cout<<name<<" wrote\n" ;

    }

// Stops the recording
    m_file->close();
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


}

void AudioProducer::single_channel_synthesis(const Tuple3ui& tri, const Vector3d& dir, const Point3d& cam, float amplitude)
{
    //// force in modal space
    double duration;
    memset( mForce_.data(), 0, sizeof(double)*mForce_.size() );
    // mForce is force in modal space
    // select a surface, then apply force to three vertices
    //*dir *= 10;
    // force is double type
    // vidS2T is vertex map for surface VID to tet VID
    // tri is a surface ID
    // tri.x is a vertex ID
    modal_->accum_modal_impulse( vidS2T_[tri.x]-numFixed_, &dir, mForce_.data() );
    modal_->accum_modal_impulse( vidS2T_[tri.y]-numFixed_, &dir, mForce_.data() );
    modal_->accum_modal_impulse( vidS2T_[tri.z]-numFixed_, &dir, mForce_.data() );

    const vector<double>& omegaD = modal_->damped_omega();
    const vector<double>& c      = modal_->damping_vector();

    // cout<<"modal_->damping_vector, c: \n";
    // for(int i = 0; i < c.size(); i++){
    //     cout << c[i];
    // }

    //// multiply with the impulse response of each mode
    memset( soundBuffer_.data(), 0, sizeof(double)*soundBuffer_.size() );
    const int totTicks = SR*TS;
    for(int i = 0;i < modal_->num_modes();++ i)
    {   

        FMMTransferEval::TComplex trans = transfer_[i]->eval( cam ); //Evaluate the sound pressure of every mode at a given position

        cout<<"moment: "<< i <<"  mForce: "<<mForce_[i]<<"\n";

        const double SS = mForce_[i] * abs(trans) / omegaD[i];

        for(int ti = 0;ti < totTicks;++ ti)
        {
            const double ts = static_cast<double>(ti) / static_cast<double>(SR);    // time
            const double amp = exp(- c[i]*0.5*ts ) ;    // exp(-xi * omega * t), damping amplitude
            if ( amp < 1E-5 ) break;
            soundBuffer_[ti] += amp * SS * sin( omegaD[i]*ts ) * amplitude;  // sin(omega_d * t)
        }
    } // end for

    // normalize the sound only for the first time, so the sound volume can change as camera moves
    if ( normalizeScale_ < 0. )
    {
        double AMP = 0;
        for(int ti = 0;ti < totTicks;++ ti)
            AMP = std::max( AMP, abs(soundBuffer_[ti]) );
        normalizeScale_ = 0.6 / AMP;
    } // end if

///////////////////////////////////////////////////////////////////////////////
    //normalizeScale_ *= amplitude;
///////////////////////////////////////////////////////////////////////////////

    // unsigned char *ptr = reinterpret_cast<unsigned char *>(buffer_.data());
    // const int channelBytes = format_.sampleSize() / 8;
    // for(int ti = 0;ti < totTicks;++ ti)
    // {
    //     const qint16 value = static_cast<qint16>( soundBuffer_[ti] * normalizeScale_ * 32767 );
    //     qToLittleEndian<qint16>(value, ptr);
    //     ptr += channelBytes;
    // }
}

void AudioProducer::generate_continuous_wav(QByteArray& buffer, vector<double>& whole_soundBuffer){
  int seq = 1;
  string name="continuous_audio1.wav";
  string name_raw="continuous_audio1.raw";
  while(exists_test(name) || exists_test(name_raw)){
    seq++;
    string name_seq = to_string(seq);
    name = "continuous_audio"+name_seq+".wav";
    name_raw = "continuous_audio"+name_seq+".raw";
  }

  QString wav_name = QString::fromStdString(name);

  WavPcmFile *m_file = new WavPcmFile(wav_name, format_);
  if(m_file->open()) {
      //audioOutput_->start(m_file);
      m_file->write(buffer);
      cout<<name<<" wrote\n" ;
  }
  m_file->close();


  ofstream rawfile;
  rawfile.open(name_raw);
  for (int i = 0; i < whole_soundBuffer.size(); ++i){
    rawfile << whole_soundBuffer.at(i) << "\n";
  }
  cout<<name_raw<<" wrote\n";
  rawfile.close();

}
