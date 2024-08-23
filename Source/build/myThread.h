#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <random>
using namespace std;
class myQFrameBuffer
{
public:
	myQFrameBuffer(void);
	myQFrameBuffer(const myQFrameBuffer& Other);
	myQFrameBuffer& myQFrameBuffer::operator=(const myQFrameBuffer& Other);
	virtual ~myQFrameBuffer(void);
	void Set(unsigned char* pPixels, const int& Width, const int& Height);
	unsigned char* GetPixels(void) { return m_pPixels; }
	int GetWidth(void) const { return m_Width; }
	int GetHeight(void) const { return m_Height; }
	int GetNoPixels(void) const { return m_NoPixels; }

	mutex			m_Mutex;

private:
	unsigned char* m_pPixels;
	int				m_Width;
	int				m_Height;
	int				m_NoPixels;
};
extern myQFrameBuffer gMyFrameBuffer;
class myQRenderThread : public thread {
public:
	myQRenderThread(const std::string& FileName = "");
	myQRenderThread(const myQRenderThread& Other);

	void run();
	virtual ~myQRenderThread(void);
	myQRenderThread& myQRenderThread::operator=(const myQRenderThread& Other);

	bool			Load(std::string& FileName);

	std::string		GetFileName(void) const { return m_FileName; }
	void			SetFileName(const std::string& FileName) { m_FileName = FileName; }
	CColorRgbLdr*   GetRenderImage(void) const;
	void			Close(void) { m_Abort = true; }
	void			PauseRendering(const bool& Pause) { m_Pause = Pause; }

private:
	std::string			m_FileName;
	CColorRgbLdr* m_pRenderImage;
	short* m_pDensityBuffer;
	short* m_pGradientMagnitudeBuffer;


public:
	bool			m_Abort;
	bool			m_Pause;
	mutex			m_Mutex;

public:
	list<int>	m_SaveFrames;
	std::string		m_SaveBaseName;

public://槽函数
	void OnUpdateTransferFunction(void);
	void OnUpdateCamera(void);
	void OnUpdateLighting(void);
	void OnRenderPause(const bool& Pause);

};


void myStartRenderThread(std::string& FileName);
void myKillRenderThread(void);

extern mutex gMySceneMutex;
extern int gMyCurrentDeviceID;
