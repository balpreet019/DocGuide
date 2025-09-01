import os
import fitz  # PyMuPDF
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class MedicalChatbot:
    def __init__(self, pdf_folder=None, csv_folder=None, persist_dir="medical_db"):
        self.pdf_folder = pdf_folder
        self.csv_folder = csv_folder
        self.persist_dir = persist_dir
        self.image_dir = os.path.join(os.path.dirname(__file__), "static", "extracted_images")
        self.table_dir = os.path.join(os.path.dirname(__file__), "static", "extracted_tables")
        self.vector_store = None
        self.qa_chain = None
        self.memory = None
        self.initialize_chatbot()

    def initialize_chatbot(self):
        # Create directories if they don't exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        text_docs = []
        
        # Process PDF files if folder exists
        if self.pdf_folder and os.path.exists(self.pdf_folder):
            pdf_files = [os.path.join(self.pdf_folder, f) 
                        for f in os.listdir(self.pdf_folder) 
                        if f.endswith(".pdf")]
            
            # Load PDF text
            for pdf_path in pdf_files:
                loader = PyPDFLoader(file_path=pdf_path)
                text_docs.extend(loader.load())
            
            # Extract Images from PDFs
            image_docs = []
            for pdf_path in pdf_files:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                doc = fitz.open(pdf_path)
                for page_index in range(len(doc)):
                    for img_index, img in enumerate(doc[page_index].get_images(full=True)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img_ext = base_image["ext"]
                        image_filename = f"{pdf_name}_page{page_index+1}_img{img_index+1}.{img_ext}"
                        image_path = os.path.join(self.image_dir, image_filename)
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        caption = f"Image from {pdf_name}, page {page_index+1}"
                        image_docs.append(Document(
                            page_content=caption,
                            metadata={
                                "source": f"/static/extracted_images/{image_filename}",
                                "type": "image",
                                "original_path": image_path
                            }
                        ))

            # Extract Tables from PDFs
            table_docs = []
            for pdf_path in pdf_files:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        tables = page.extract_tables()
                        for idx, table in enumerate(tables):
                            if table:
                                table_filename = f"{pdf_name}_page{page_num}_table{idx+1}.txt"
                                table_path = os.path.join(self.table_dir, table_filename)
                                table_str = "\n".join([
                                    "\t".join([str(cell) if cell is not None else "" for cell in row])
                                    for row in table 
                                ])
                                with open(table_path, "w", encoding="utf-8") as f:
                                    f.write(table_str)
                                caption = f"Table from {pdf_name}, page {page_num}"
                                table_docs.append(Document(
                                    page_content=caption,
                                    metadata={
                                        "source": f"/static/extracted_tables/{table_filename}",
                                        "type": "table",
                                        "original_path": table_path
                                    }
                                ))
        else:
            image_docs = []
            table_docs = []

        # Process CSV files if folder exists - USING CSVLoader NOW
        if self.csv_folder and os.path.exists(self.csv_folder):
            csv_files = [os.path.join(self.csv_folder, f) 
                        for f in os.listdir(self.csv_folder) 
                        if f.endswith(".csv")]
            
            for csv_path in csv_files:
                try:
                    # Configure CSVLoader with appropriate parameters
                    loader = CSVLoader(
                        file_path=csv_path,
                        encoding='utf-8',
                        csv_args={
                            'delimiter': ',',
                            'quotechar': '"',
                            'fieldnames': None  # Auto-detect headers
                        }
                    )
                    docs = loader.load()
                    
                    # Add metadata to identify CSV sources and preserve original data
                    for doc in docs:
                        # Extract relevant fields from content if needed
                        content = doc.page_content
                        doc.metadata.update({
                            "source": csv_path,
                            "type": "csv_row",
                            "source_type": "csv"
                        })
                        
                        # You can parse the content to extract specific fields if needed
                        # Example (adjust based on your CSV structure):
                        if "Name:" in content and "Specialty:" in content:
                            parts = content.split("\n")
                            name = parts[0].replace("Name:", "").strip()
                            specialty = parts[1].replace("Specialty:", "").strip()
                            location = parts[2].replace("Location:", "").strip() if len(parts) > 2 else "N/A"
                            
                            doc.metadata.update({
                                "name": name,
                                "specialty": specialty,
                                "location": location
                            })
                    
                    text_docs.extend(docs)
                except Exception as e:
                    print(f"Error processing CSV {csv_path}: {str(e)}")

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Optimal for medical data
            chunk_overlap=150,
            separators=["\n\n", "\n", "  ", ""]  # Preserve structure
        )
        text_chunks = splitter.split_documents(text_docs)
        
        # Combine all documents
        all_docs = text_chunks
        if self.pdf_folder:
            all_docs += image_docs + table_docs
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=self.persist_dir
        )
        
        # Initialize LLM
        llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
        
        # Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Enhanced prompt template for CSV data
        doctor_prompt = PromptTemplate.from_template("""
        You are a knowledgeable medical chatbot. When answering:
        
        1. For doctor queries, include:
           - Name (if available)
           - Specialty (if available)
           - Hospital/Location (if available)
           - Additional details from source
        
        2. For medical questions, provide authoritative answers.
        
        3. Reference images/tables when relevant:
           [image: description](image_url)
           [table: description](table_url)

        **Always format your answers in a structured way using bullet points, numbered lists, or tables where appropriate. Do not write long paragraphs.**

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)
        
        # QA Chain with metadata filtering
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 5
                }
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": doctor_prompt}
        )
    
    def get_response(self, question):
        result = self.qa_chain.invoke({"question": question})
        
        # Process sources
        sources = {"text": [], "images": [], "tables": [], "doctors": []}
        
        for doc in result['source_documents']:
            source_type = doc.metadata.get("type", "text")
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            
            if source_type == "image":
                sources["images"].append(source_info)
            elif source_type == "table":
                sources["tables"].append(source_info)
            elif source_type == "csv_row":
                sources["doctors"].append(source_info)
            else:
                sources["text"].append(source_info)
        
        return {
            "answer": result['answer'],
            "sources": sources
        }