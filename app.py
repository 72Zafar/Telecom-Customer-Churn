from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipelines models from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import ChurnDataClassifer, ChurnData
from src.pipline.training_pipeline import TrainingPipeline

# Initializing the Fastapi application
app = FastAPI()

# Mount the 'frontend_web' directory for serving static files (like csv)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates engine for rendering streamlit apps
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resourcs Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """

    def __init__(self, request:Request):
        self.request:Request = request
        self.Gender: Optional[int] = None
        self.Married: Optional[int] = None
        self.Offer: Optional[int] = None
        self.Phone_Service: Optional[int] = None
        self.Multiple_Lines: Optional[int] = None
        self.Internet_Service: Optional[int] = None
        self.Internet_Type: Optional[int] = None
        self.Online_Security: Optional[int] = None
        self.Online_Backup: Optional[int] = None
        self.Device_Protection_Plan: Optional[int] = None
        self.Premium_Tech_Support: Optional[int] = None
        self.Streaming_TV: Optional[int] = None
        self.Streaming_Movies: Optional[int] = None
        self.Streaming_Music: Optional[int] = None
        self.Unlimited_Data: Optional[int] = None
        self.Contract: Optional[int] = None
        self.Paperless_Billing: Optional[int] = None
        self.Payment_Method: Optional[int] = None
        self.Age: Optional[int] = None
        self.Number_of_Dependents: Optional[int] = None
        self.Number_of_Referrals: Optional[int] = None
        self.Tenure_in_Months: Optional[int] = None
        self.Avg_Monthly_Long_Distance_Charges: Optional[float] = None
        self.Avg_Monthly_GB_Download: Optional[float] = None
        self.Monthly_Charge: Optional[float] = None
        self.Total_Charges: Optional[float] = None
        
    async def get_churn_data(self):
        """
        Method to retrieve and assign from data to class attibutes.
        This method is asynchronous to handle from data fetching without blocking.
        """
        form  = await self.request.form()
        self.Gender = form.get("Gender")
        self.Married = form.get("Married")
        self.Offer = form.get("Offer")
        self.Phone_Service = form.get("Phone_Service")
        self.Multiple_Lines = form.get("Multiple_Lines")
        self.Internet_Service = form.get("Internet_Service")
        self.Internet_Type = form.get("Internet_Type")
        self.Online_Security = form.get("Online_Security")
        self.Online_Backup = form.get("Online_Backup")
        self.Device_Protection_Plan = form.get("Device_Protection_Plan")
        self.Premium_Tech_Support = form.get("Premium_Tech_Support")
        self.Streaming_TV = form.get("Streaming_TV")
        self.Streaming_Movies = form.get("Streaming_Movies")
        self.Streaming_Music = form.get("Streaming_Music")
        self.Unlimited_Data = form.get("Unlimited_Data")
        self.Contract = form.get("Contract")
        self.Paperless_Billing = form.get("Paperless_Billing")
        self.Payment_Method = form.get("Payment_Method")
        self.Age = form.get("Age")
        self.Number_of_Dependents = form.get("Number_of_Dependents")
        self.Number_of_Referrals = form.get("Number_of_Referrals")
        self.Tenure_in_Months = form.get("Tenure_in_Months")
        self.Avg_Monthly_Long_Distance_Charges = form.get("Avg_Monthly_Long_Distance_Charges")
        self.Avg_Monthly_GB_Download = form.get("Avg_Monthly_GB_Download")
        self.Monthly_Charge = form.get("Monthly_Charge")
        self.Total_Charges = form.get("Total_Charges")

        pass


# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main page for the churn data input.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline
    """
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error during training: {str(e)}")

# Route to handle form submission and make predictions
@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_churn_data()

        churn_obj = ChurnData(
            Gender=form.Gender,
            Married=form.Married,
            Offer=form.Offer,
            Phone_Service=form.Phone_Service,
            Multiple_Lines=form.Multiple_Lines,
            Internet_Service=form.Internet_Service,
            Internet_Type=form.Internet_Type,
            Online_Security=form.Online_Security,
            Online_Backup=form.Online_Backup,
            Device_Protection_Plan=form.Device_Protection_Plan,
            Premium_Tech_Support=form.Premium_Tech_Support,
            Streaming_TV=form.Streaming_TV,
            Streaming_Movies=form.Streaming_Movies,
            Streaming_Music=form.Streaming_Music,
            Unlimited_Data=form.Unlimited_Data,
            Contract=form.Contract,
            Paperless_Billing=form.Paperless_Billing,
            Payment_Method=form.Payment_Method,
            Age=form.Age,
            Number_of_Dependents=form.Number_of_Dependents,
            Number_of_Referrals=form.Number_of_Referrals,
            Tenure_in_Months=form.Tenure_in_Months,
            Avg_Monthly_Long_Distance_Charges=form.Avg_Monthly_Long_Distance_Charges,
            Avg_Monthly_GB_Download=form.Avg_Monthly_GB_Download,
            Monthly_Charge=form.Monthly_Charge,
            Total_Charges=form.Total_Charges,
        )

        # Convert form data into a dictionary for the model
        churn_dict = churn_obj.get_churn_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = ChurnDataClassifer()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=churn_dict)[0]

        # Return JSON response
        if value == 1:
            return {"status": True, "prediction": "Churn", "message": "Customer is Likely to Churn"}
        else:
            return {"status": True, "prediction": "Stay", "message": "Customer Will Stay"}
    except Exception as e:
        return {"status": False, "prediction": None, "message": f'Error: {str(e)}'}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)