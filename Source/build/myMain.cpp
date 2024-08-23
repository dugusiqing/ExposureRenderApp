#include <iostream>
#include <vtkErrorCode.h>
#include <vtkImageAccumulate.h>
#include <vtkImageCast.h>
#include <vtkMetaImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageGradientMagnitude.h>

#include "Histogram.h"
#include "Scene.h"
#include <myMain.h>
#include <vtkVersionMacros.h>

#include "Core.cuh"
#include "MainWindow.h"
using namespace std;

void ParseTransferFunction(std::string xmlFileName, /*QList<QTransferFunction>& presetsTransferFunction,*/std::string selectedPreset= "manix_small")
{
	QList<QTransferFunction> presetsTransferFunction;

	QFile XmlFile;
	XmlFile.setFileName(QString::fromStdString(xmlFileName));
	// Open the XML file for reading
	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		std::cout << "Open  XML failed！" << std::endl;
		return;
	}
	// Document object model for XML
	QDomDocument DOM;
	// ParseXML file content into DOM
	if (!DOM.setContent(&XmlFile))
	{
		std::cout << "Failed to parse " + xmlFileName + ".xml into a DOM tree." << std::endl;
		XmlFile.close();
		return;
	}
	// Obtain document root node
	QDomElement Root = DOM.documentElement();
	QDomNodeList Presets = Root.elementsByTagName("Preset");
	for (int i = 0; i < Presets.count(); i++)
	{
		QDomNode Node = Presets.item(i);


		//toElement().tagName()
		QString nodeName = Node.attributes().item(0).nodeValue();
		// Append the transfer function
		if (Node.attributes().item(0).nodeValue() == QString::fromStdString(selectedPreset))
		{
			QTransferFunction NewPreset;
			presetsTransferFunction.append(NewPreset);
			// Load the preset into it
			presetsTransferFunction.back().ReadXML(Node.toElement()); break;
		}
		//continue;

	}
	XmlFile.close();
	QTransferFunction TransferFunction = presetsTransferFunction.at(0);
	gScene.m_TransferFunctions.m_Opacity.m_NoNodes = TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Diffuse.m_NoNodes = TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Specular.m_NoNodes = TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Emission.m_NoNodes = TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Roughness.m_NoNodes = TransferFunction.GetNodes().size();

	for (int i = 0; i < TransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = TransferFunction.GetNode(i);

		const float Intensity = Node.GetIntensity();

		// Positions
		gScene.m_TransferFunctions.m_Opacity.m_P[i] = Intensity;
		gScene.m_TransferFunctions.m_Diffuse.m_P[i] = Intensity;
		gScene.m_TransferFunctions.m_Specular.m_P[i] = Intensity;
		gScene.m_TransferFunctions.m_Emission.m_P[i] = Intensity;
		gScene.m_TransferFunctions.m_Roughness.m_P[i] = Intensity;

		// Colors
		gScene.m_TransferFunctions.m_Opacity.m_C[i] = CColorRgbHdr(Node.GetOpacity());
		gScene.m_TransferFunctions.m_Diffuse.m_C[i] = CColorRgbHdr(Node.GetDiffuse().redF(), Node.GetDiffuse().greenF(), Node.GetDiffuse().blueF());
		gScene.m_TransferFunctions.m_Specular.m_C[i] = CColorRgbHdr(Node.GetSpecular().redF(), Node.GetSpecular().greenF(), Node.GetSpecular().blueF());
		gScene.m_TransferFunctions.m_Emission.m_C[i] = 500.0f * CColorRgbHdr(Node.GetEmission().redF(), Node.GetEmission().greenF(), Node.GetEmission().blueF());

		const float Roughness = 1.0f - expf(-Node.GetGlossiness());

		gScene.m_TransferFunctions.m_Roughness.m_C[i] = CColorRgbHdr(Roughness * 250.0f);
	}

	gScene.m_DensityScale = TransferFunction.GetDensityScale();
	gScene.m_ShadingType = TransferFunction.GetShadingType();
	gScene.m_GradientFactor = TransferFunction.GetGradientFactor();
	gScene.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void ParseCamera(std::string xmlFileName,/*QList<QCamera>& presetsQCamera, */std::string selectedPreset = "manix_small")
{

	QList<QCamera> presetsQCamera;

	QFile XmlFile;
	XmlFile.setFileName(QString::fromStdString(xmlFileName));
	// Open the XML file for reading
	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		std::cout << "Open  XML failed！" << std::endl;
		return;
	}
	// Document object model for XML
	QDomDocument DOM;
	// ParseXML file content into DOM
	if (!DOM.setContent(&XmlFile))
	{
		std::cout << "Failed to parse " + xmlFileName + ".xml into a DOM tree." << std::endl;
		XmlFile.close();
		return;
	}
	// Obtain document root node
	QDomElement Root = DOM.documentElement();
	QDomNodeList Presets = Root.elementsByTagName("Preset");
	for (int i = 0; i < Presets.count(); i++)
	{
		QDomNode Node = Presets.item(i);
		//toElement().tagName()
		QString nodeName = Node.attributes().item(0).nodeValue();
		// Append the transfer function
		if (Node.attributes().item(0).nodeValue() == QString::fromStdString(selectedPreset))
		{
			QCamera NewPreset;
			presetsQCamera.append(NewPreset);
			presetsQCamera.back().ReadXML(Node.toElement());
			break;
		}
	}
	XmlFile.close();
	QCamera tmpCamera= presetsQCamera.at(0);
	gScene.m_Camera.m_Film.m_Exposure = 1.0f - tmpCamera.GetFilm().GetExposure();
	if (tmpCamera.GetFilm().IsDirty())
	{
		const int FilmWidth = tmpCamera.GetFilm().GetWidth();
		const int FilmHeight = tmpCamera.GetFilm().GetHeight();

		gScene.m_Camera.m_Film.m_Resolution.SetResX(FilmWidth);
		gScene.m_Camera.m_Film.m_Resolution.SetResY(FilmHeight);
		gScene.m_Camera.Update();
		tmpCamera.GetFilm().UnDirty();
		// 		// 
		gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}
	gScene.m_Camera.m_From	= tmpCamera.GetFrom();
	gScene.m_Camera.m_Target	= tmpCamera.GetTarget();
	gScene.m_Camera.m_Up		= tmpCamera.GetUp();
	gScene.m_Camera.Update();
	// Aperture
	gScene.m_Camera.m_Aperture.m_Size = tmpCamera.GetAperture().GetSize();
	// Projection
	gScene.m_Camera.m_FovV = tmpCamera.GetProjection().GetFieldOfView();
	// Focus
	gScene.m_Camera.m_Focus.m_Type = (CFocus::EType)tmpCamera.GetFocus().GetType();
	gScene.m_Camera.m_Focus.m_FocalDistance = tmpCamera.GetFocus().GetFocalDistance();

	gScene.m_DenoiseParams.m_Enabled = tmpCamera.GetFilm().GetNoiseReduction();
	gScene.m_DirtyFlags.SetFlag(CameraDirty);
}





void ParseLight(std::string xmlFileName,/*QList<QCamera>& presetsQCamera, */std::string selectedPreset = "manix_small")
{
	QList<QLighting> presetsQLight;

	QFile XmlFile;
	XmlFile.setFileName(QString::fromStdString(xmlFileName));
	// Open the XML file for reading
	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		std::cout << "Open  XML failed！" << std::endl;
		return;
	}
	// Document object model for XML
	QDomDocument DOM;
	// ParseXML file content into DOM
	if (!DOM.setContent(&XmlFile))
	{
		std::cout << "Failed to parse " + xmlFileName + ".xml into a DOM tree." << std::endl;
		XmlFile.close();
		return;
	}
	// Obtain document root node
	QDomElement Root = DOM.documentElement();
	QDomNodeList Presets = Root.elementsByTagName("Preset");
	for (int i = 0; i < Presets.count(); i++)
	{
		QDomNode Node = Presets.item(i);
		//toElement().tagName()
		QString nodeName = Node.attributes().item(0).nodeValue();
		// Append the transfer function
		if (Node.attributes().item(0).nodeValue() == QString::fromStdString(selectedPreset))
		{
			QLighting NewPreset;
			presetsQLight.append(NewPreset);
			presetsQLight.back().ReadXML(Node.toElement());
			break;
		}
	}
	XmlFile.close();
	QLighting tmpLight = presetsQLight.at(0);

	gScene.m_Lighting.Reset();

	if (tmpLight.Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_T = 1;

		BackgroundLight.m_ColorTop = tmpLight.Background().GetIntensity() * CColorRgbHdr(tmpLight.Background().GetTopColor().redF(), tmpLight.Background().GetTopColor().greenF(), tmpLight.Background().GetTopColor().blueF());
		BackgroundLight.m_ColorMiddle = tmpLight.Background().GetIntensity() * CColorRgbHdr(tmpLight.Background().GetMiddleColor().redF(), gLighting.Background().GetMiddleColor().greenF(), tmpLight.Background().GetMiddleColor().blueF());
		BackgroundLight.m_ColorBottom = tmpLight.Background().GetIntensity() * CColorRgbHdr(tmpLight.Background().GetBottomColor().redF(), gLighting.Background().GetBottomColor().greenF(), tmpLight.Background().GetBottomColor().blueF());

		BackgroundLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(BackgroundLight);
	}

	for (int i = 0; i < tmpLight.GetLights().size(); i++)
	{
		QLight& Light = tmpLight.GetLights()[i];

		CLight AreaLight;

		AreaLight.m_T = 0;
		AreaLight.m_Theta = Light.GetTheta() / RAD_F;
		AreaLight.m_Phi = Light.GetPhi() / RAD_F;
		AreaLight.m_Width = Light.GetWidth();
		AreaLight.m_Height = Light.GetHeight();
		AreaLight.m_Distance = Light.GetDistance();
		AreaLight.m_Color = Light.GetIntensity() * CColorRgbHdr(Light.GetColor().redF(), Light.GetColor().greenF(), Light.GetColor().blueF());

		AreaLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(AreaLight);
	}

	gScene.m_DirtyFlags.SetFlag(LightsDirty);
}


void SetUpWindow(QVTKOpenGLNativeWidget &QtVtkWidget)
{
	vtkSmartPointer<vtkImageActor>				m_ImageActor;
	vtkSmartPointer<vtkImageImport>				m_ImageImport;
	vtkSmartPointer<vtkInteractorStyleImage>	m_InteractorStyleImage;
	vtkSmartPointer<vtkRenderer>				m_SceneRenderer;
	vtkSmartPointer<vtkRenderWindow>			m_RenderWindow;
	vtkSmartPointer<vtkRenderWindowInteractor>	m_RenderWindowInteractor;
	vtkSmartPointer<vtkCallbackCommand>			m_KeyPressCallback;
	vtkSmartPointer<vtkCallbackCommand>			m_KeyReleaseCallback;
	vtkSmartPointer<vtkRealisticCameraStyle>	m_InteractorStyleRealisticCamera;
	m_SceneRenderer = vtkRenderer::New();
	m_SceneRenderer->SetBackground(0.25, 0.25, 0.25);
	m_SceneRenderer->SetBackground2(0.25, 0.25, 0.25);
	m_SceneRenderer->SetGradientBackground(true);
	m_SceneRenderer->GetActiveCamera()->SetPosition(0.0, 0.0, 1.0);
	m_SceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
	m_SceneRenderer->GetActiveCamera()->ParallelProjectionOn();
	m_RenderWindow = QtVtkWidget.renderWindow();
	m_RenderWindow->AddRenderer(m_SceneRenderer);
	
	m_InteractorStyleRealisticCamera = vtkSmartPointer<vtkRealisticCameraStyle>::New();
	m_InteractorStyleImage = vtkInteractorStyleImage::New();
	
	m_RenderWindow->GetInteractor()->SetInteractorStyle(m_InteractorStyleRealisticCamera);

	m_RenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, m_KeyPressCallback);
	m_RenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyReleaseEvent, m_KeyReleaseCallback);
	m_ImageImport = vtkImageImport::New();
	m_ImageActor = vtkImageActor::New();
}

int main1()
{
	std::string presetName = "manix_small";
	std::string VolumeFile = R"(D:\Works\code\ExposureRenderApp\Source\build\Release\Examples\)"+ presetName+".mhd";


	//加载Preset
	std::string xmlFileName = R"(D:\Works\code\ExposureRenderApp\Source\build\Release\AppearancePresets.xml)";
	ParseTransferFunction(xmlFileName, presetName);

	xmlFileName = R"(D:\Works\code\ExposureRenderApp\Source\build\Release\CameraPresets.xml)";
	ParseCamera(xmlFileName, presetName);

	xmlFileName = R"(D:\Works\code\ExposureRenderApp\Source\build\Release\LightingPresets.xml)";
	ParseLight(xmlFileName, presetName);


	QVTKOpenGLNativeWidget QtVtkWidget;
	SetUpWindow(QtVtkWidget);





	//设置显卡
	int CudaDeviceID = 0;
	const cudaError_t CudaError = cudaSetDevice(CudaDeviceID);
	//加载Volume
	if (CudaError != cudaSuccess)
	{
		std::cout << "cudaSetDevice:NG" << std::endl;
	}
	else
	{
		std::cout << "cudaSetDevice:OK" << std::endl;
	}
	//加载Volume文件，读取Scene
	//内部含有解析xml的Qt代码D:\\Works\\code\\ExposureRenderApp\\Source\\build\\Release\\Examples\\manix_small.mhd
	//std::string VolumeFile=R"(D:\Works\code\ExposureRenderApp\Source\build\Release\Examples\manix_small.mhd)";
	myLoad(QString::fromStdString(VolumeFile));







	CScene SceneCopy;

	gScene.m_Camera.m_SceneBoundingBox = gScene.m_BoundingBox;
	gScene.m_Camera.SetViewMode(ViewModeFront);
	gScene.m_Camera.Update();
	gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	cudaExtent Res;
	Res.width = gScene.m_Resolution[0];
	Res.height = gScene.m_Resolution[1];
	Res.depth = gScene.m_Resolution[2];

	BindDensityBuffer((short*)m_pDensityBuffer, Res);
	BindGradientMagnitudeBuffer((short*)m_pGradientMagnitudeBuffer, Res);
	ResetRenderCanvasView();

	CTiming FPS, RenderImage, BlurImage, PostProcessImage, DenoiseImage;

	{
		try
		{
			while ('c' != getchar())
			{
				/*if (m_Pause)
					continue;*/

					/*gStatus.SetPreRenderFrame();*/

					// CUDA time for profiling
				CCudaTimer TmrFps;

				SceneCopy = gScene;

				/*gStatus.SetStatisticChanged("Camera", "Position", FormatVector(SceneCopy.m_Camera.m_From));
				gStatus.SetStatisticChanged("Camera", "Target", FormatVector(SceneCopy.m_Camera.m_Target));
				gStatus.SetStatisticChanged("Camera", "Up Vector", FormatVector(SceneCopy.m_Camera.m_Up));*/

				// Resizing the image canvas requires special attention
				if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
				{
					if (!m_pRenderImage)
					{
						free(m_pRenderImage);
						m_pRenderImage = NULL;
					}
					m_pRenderImage = (CColorRgbLdr*)malloc(SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
					if (m_pRenderImage)
						memset(m_pRenderImage, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
					SceneCopy.SetNoIterations(0);
				}

				// Restart the rendering when when the camera, lights and render params are dirty
				if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
				{
					ResetRenderCanvasView();
					// Reset no. iterations
					gScene.SetNoIterations(0);
				}

				// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
				gScene.m_DirtyFlags.ClearAllFlags();

				SceneCopy.m_DenoiseParams.SetWindowRadius(3.0f);
				SceneCopy.m_DenoiseParams.m_LerpC = 0.33f * (max((float)gScene.GetNoIterations(), 1.0f) * 0.035f);
				SceneCopy.m_Camera.Update();

				BindConstants(&SceneCopy);

				BindTransferFunctionOpacity(SceneCopy.m_TransferFunctions.m_Opacity);
				BindTransferFunctionDiffuse(SceneCopy.m_TransferFunctions.m_Diffuse);
				BindTransferFunctionSpecular(SceneCopy.m_TransferFunctions.m_Specular);
				BindTransferFunctionRoughness(SceneCopy.m_TransferFunctions.m_Roughness);
				BindTransferFunctionEmission(SceneCopy.m_TransferFunctions.m_Emission);

				BindRenderCanvasView(SceneCopy.m_Camera.m_Film.m_Resolution);

				Render(0, SceneCopy, RenderImage, BlurImage, PostProcessImage, DenoiseImage);

				gScene.SetNoIterations(gScene.GetNoIterations() + 1);

				gStatus.SetStatisticChanged("Timings", "Render Image", QString::number(RenderImage.m_FilteredDuration, 'f', 2), "ms.");
				gStatus.SetStatisticChanged("Timings", "Blur Estimate", QString::number(BlurImage.m_FilteredDuration, 'f', 2), "ms.");
				gStatus.SetStatisticChanged("Timings", "Post Process Estimate", QString::number(PostProcessImage.m_FilteredDuration, 'f', 2), "ms.");
				gStatus.SetStatisticChanged("Timings", "De-noise Image", QString::number(DenoiseImage.m_FilteredDuration, 'f', 2), "ms.");

				FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

				gStatus.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
				gStatus.SetStatisticChanged("Performance", "No. Iterations", QString::number(SceneCopy.GetNoIterations()), "Iterations");

				HandleCudaError(cudaMemcpy(m_pRenderImage, GetDisplayEstimate(), SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr), cudaMemcpyDeviceToHost));

				gFrameBuffer.Set((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.GetWidth(), SceneCopy.m_Camera.m_Film.GetHeight());
				const QString ImageFilePath = /*QApplication::applicationDirPath() + */"./Output/" + m_SaveBaseName + "_" + QString::number(SceneCopy.GetNoIterations()) + ".png";
				SaveImage((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY(), ImageFilePath);
				if (m_SaveFrames.indexOf(SceneCopy.GetNoIterations()) > 0)
				{


					SaveImage((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY(), ImageFilePath);
				}

				gStatus.SetPostRenderFrame();
			}
		}
		catch (QString* pMessage)
		{
			//		Log(*pMessage + ", rendering will be aborted");

			free(m_pRenderImage);
			m_pRenderImage = NULL;

			gStatus.SetRenderEnd();

			return -1;
		}
		free(m_pRenderImage);
		m_pRenderImage = NULL;
		UnbindDensityBuffer();
		UnbindTransferFunctionOpacity();
		UnbindTransferFunctionDiffuse();
		UnbindTransferFunctionSpecular();
		UnbindTransferFunctionRoughness();
		UnbindTransferFunctionEmission();
		FreeRenderCanvasView();
		// Clear the histogram
		gHistogram.Reset();

		ResetDevice();
	}
	getchar();
	return 1;
}

bool myLoad(QString& FileName)
{
	std::cout << "myLoad Begin" << std::endl;
	m_FileName = FileName;

	// Create meta image reader
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	QFileInfo FileInfo(FileName);

	if (!FileInfo.exists())
	{
		Log(QString(QFileInfo(FileName).filePath().replace("//", "/")).toLatin1() + "  does not exist!", QLogger::Critical);
		return false;
	}

	Log(QString("Loading " + QFileInfo(FileName).fileName()).toLatin1());

	// Exit if the reader can't read the file
	if (!MetaImageReader->CanReadFile(m_FileName.toLatin1()))
	{
		Log(QString("Meta image reader can't read file " + QFileInfo(FileName).fileName()).toLatin1(), QLogger::Critical);
		return false;
	}

	MetaImageReader->SetFileName(m_FileName.toLatin1());

	MetaImageReader->Update();

	if (MetaImageReader->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("Error loading file " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}

	vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();

	Log("Casting volume data type to short", "grid");

	ImageCast->SetInputConnection(MetaImageReader->GetOutputPort());
	ImageCast->SetOutputScalarTypeToShort();
	ImageCast->Update();

	if (ImageCast->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("vtkImageCast error: " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}

	// Volume resolution
	int* pVolumeResolution = ImageCast->GetOutput()->GetExtent();
	gScene.m_Resolution.SetResXYZ(Vec3i(pVolumeResolution[1] + 1, pVolumeResolution[3] + 1, pVolumeResolution[5] + 1));

	Log("Resolution: " + FormatSize(gScene.m_Resolution.GetResXYZ()) + "", "grid");

	// Intensity range
	double* pIntensityRange = ImageCast->GetOutput()->GetScalarRange();
	gScene.m_IntensityRange.SetMin((float)pIntensityRange[0]);
	gScene.m_IntensityRange.SetMax((float)pIntensityRange[1]);

	Log("Intensity range: [" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "grid");

	// Spacing
	double* pSpacing = ImageCast->GetOutput()->GetSpacing();

	gScene.m_Spacing.x = (float)pSpacing[0];
	gScene.m_Spacing.y = (float)pSpacing[1];
	gScene.m_Spacing.z = (float)pSpacing[2];

	Log("Spacing: " + FormatSize(gScene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(gScene.m_Spacing.x * (float)gScene.m_Resolution.GetResX(), gScene.m_Spacing.y * (float)gScene.m_Resolution.GetResY(), gScene.m_Spacing.z * (float)gScene.m_Resolution.GetResZ()));

	// Compute the volume's bounding box
	gScene.m_BoundingBox.m_MinP = Vec3f(0.0f);
	gScene.m_BoundingBox.m_MaxP = PhysicalSize / PhysicalSize.Max();

	gScene.m_GradientDelta = 1.0f / (float)gScene.m_Resolution.GetMax();

	Log("Bounding box: " + FormatVector(gScene.m_BoundingBox.m_MinP, 2) + " - " + FormatVector(gScene.m_BoundingBox.m_MaxP), "grid");

	const int DensityBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);

	m_pDensityBuffer = (short*)malloc(DensityBufferSize);
	memcpy(m_pDensityBuffer, ImageCast->GetOutput()->GetScalarPointer(), DensityBufferSize);

	// Gradient magnitude volume
	vtkSmartPointer<vtkImageGradientMagnitude> GradientMagnitude = vtkImageGradientMagnitude::New();

	Log("Creating gradient magnitude volume", "grid");

	GradientMagnitude->SetDimensionality(3);
	GradientMagnitude->SetInputConnection(ImageCast->GetOutputPort());
	GradientMagnitude->Update();

	vtkImageData* GradientMagnitudeBuffer = GradientMagnitude->GetOutput();

	// Scalar range of the gradient magnitude
	double* pGradientMagnitudeRange = GradientMagnitudeBuffer->GetScalarRange();

	gScene.m_GradientMagnitudeRange.SetMin((float)pGradientMagnitudeRange[0]);
	gScene.m_GradientMagnitudeRange.SetMax((float)pGradientMagnitudeRange[1]);

	Log("Gradient magnitude range: [" + QString::number(gScene.m_GradientMagnitudeRange.GetMin(), 'f', 2) + " - " + QString::number(gScene.m_GradientMagnitudeRange.GetMax(), 'f', 2) + "]", "grid");

	const int GradientMagnitudeBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);

	m_pGradientMagnitudeBuffer = (short*)malloc(GradientMagnitudeBufferSize);
	memcpy(m_pGradientMagnitudeBuffer, GradientMagnitudeBuffer->GetScalarPointer(), GradientMagnitudeBufferSize);

	// Build the histogram
	Log("Creating gradient magnitude histogram", "grid");

	vtkSmartPointer<vtkImageAccumulate> GradMagHistogram = vtkSmartPointer<vtkImageAccumulate>::New();

	GradMagHistogram->SetInputConnection(GradientMagnitude->GetOutputPort());
	GradMagHistogram->SetComponentExtent(0, 255, 0, 0, 0, 0);
	GradMagHistogram->SetComponentOrigin(0, 0, 0);
	GradMagHistogram->SetComponentSpacing(gScene.m_GradientMagnitudeRange.GetRange() / 256.0f, 0, 0);
	//	GradMagHistogram->IgnoreZeroOn();
	GradMagHistogram->Update();

	gScene.m_GradMagMean = (float)GradMagHistogram->GetMean()[0];
	gScene.m_GradientFactor = gScene.m_GradMagMean;

	Log("Mean gradient magnitude: " + QString::number(gScene.m_GradMagMean, 'f', 2), "grid");

	Log("Creating density histogram", "grid");

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();

	Log("Creating histogram", "grid");

	Histogram->SetInputConnection(ImageCast->GetOutputPort());
	Histogram->SetComponentExtent(0, 256, 0, 0, 0, 0);
	Histogram->SetComponentOrigin(gScene.m_IntensityRange.GetMin(), 0, 0);
	Histogram->SetComponentSpacing(gScene.m_IntensityRange.GetRange() / 256.0f, 0, 0);
	Histogram->IgnoreZeroOn();
	Histogram->Update();

	// Update the histogram in the transfer function
	gHistogram.SetBins((int*)Histogram->GetOutput()->GetScalarPointer(), 256);


	////屏蔽Qt的状态刷新
	//gStatus.SetStatisticChanged("Volume", "File", QFileInfo(m_FileName).fileName(), "");
	//gStatus.SetStatisticChanged("Volume", "Bounding Box", "", "");
	//gStatus.SetStatisticChanged("Bounding Box", "Min", FormatVector(gScene.m_BoundingBox.m_MinP, 2), "m");
	//gStatus.SetStatisticChanged("Bounding Box", "Max", FormatVector(gScene.m_BoundingBox.m_MaxP, 2), "m");
	//gStatus.SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
	//gStatus.SetStatisticChanged("Volume", "Resolution", FormatSize(gScene.m_Resolution.GetResXYZ()), "Voxels");
	//gStatus.SetStatisticChanged("Volume", "Spacing", FormatSize(gScene.m_Spacing, 2), "mm");
	//gStatus.SetStatisticChanged("Volume", "No. Voxels", QString::number(gScene.m_Resolution.GetNoElements()), "Voxels");
	//gStatus.SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");
	std::cout << "myLoad End" << std::endl;
	return true;
}