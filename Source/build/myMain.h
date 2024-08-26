#pragma once
class MyQFrameBuffer
{
public:
	MyQFrameBuffer(void);
	MyQFrameBuffer(const MyQFrameBuffer& Other);
	MyQFrameBuffer& MyQFrameBuffer::operator=(const MyQFrameBuffer& Other);
	virtual ~MyQFrameBuffer(void);
	void Set(unsigned char* pPixels, const int& Width, const int& Height);
	unsigned char* GetPixels(void) { return m_pPixels; }
	int GetWidth(void) const { return m_Width; }
	int GetHeight(void) const { return m_Height; }
	int GetNoPixels(void) const { return m_NoPixels; }
	//QMutex			m_Mutex;
private:
	unsigned char* m_pPixels;
	int				m_Width;
	int				m_Height;
	int				m_NoPixels;
};
extern MyQFrameBuffer gMyFrameBuffer;
bool myLoad(QString& FileName);
QString m_FileName;
short* m_pDensityBuffer = nullptr;
short* m_pGradientMagnitudeBuffer = nullptr;
CColorRgbLdr* m_pRenderImage = nullptr;
QList<int>		m_SaveFrames;
QString			m_SaveBaseName;