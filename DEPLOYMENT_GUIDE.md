# 🚀 Railway Deployment Guide - JAI GURU DEV AI

## Quick Deploy (1-Click)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

## Manual Deployment Steps

### 1. 🌟 Prerequisites
- GitHub account
- Railway account ([railway.app](https://railway.app))
- OpenAI API key (required)
- Groq API key (optional, for faster responses)

### 2. 🔗 Connect Repository
1. Fork this repository to your GitHub
2. Go to [Railway](https://railway.app)
3. Click "Start a New Project"
4. Select "Deploy from GitHub repo"
5. Choose your forked repository

### 3. ⚙️ Set Environment Variables

In Railway dashboard, go to **Variables** tab and add:

```bash
# REQUIRED - OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# OPTIONAL - Groq API Key (for faster responses)
GROQ_API_KEY=your_groq_api_key_here

# Application Settings
ENVIRONMENT=production
```

### 4. 🚀 Deploy

Railway will automatically:
- Detect the `railway.toml` configuration
- Install Python dependencies
- Build and deploy the application
- Provide a public URL

## 🎯 Expected Performance

| Phase | Duration | Description |
|-------|----------|-------------|
| Build | 2-5 min | Installing dependencies |
| First Run | 30-60 sec | Creating vector database |
| Subsequent | 2-5 sec | Fast loading with persistence |
| Response Time | 1-3 sec | Spiritual guidance delivery |

## 🔍 Monitoring Your Deployment

### Railway Dashboard
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, network usage
- **Health**: Automatic health checks at `/_stcore/health`
- **Variables**: Environment configuration

### Success Indicators
Look for these in the logs:
```
✅ RAG System initialized successfully!
✅ Loaded existing ChromaDB with X documents
✅ Custom retriever created successfully
✅ QA chain setup complete
```

## 🛠️ Troubleshooting

### Common Issues

**1. Build Failures**
- Check Python version compatibility
- Verify requirements.txt syntax
- Review build logs

**2. App Won't Start**
- Ensure OPENAI_API_KEY is set
- Check Railway Variables tab
- Verify port configuration ($PORT)

**3. Database Issues**
- First run takes longer (database creation)
- Subsequent runs are fast (persistence)
- Check disk space in Railway dashboard

**4. API Errors**
- Verify API keys are valid
- Check API quotas and limits
- Test API connections

### Debug Commands

Access Railway shell for debugging:
```bash
# In Railway dashboard -> Settings -> Environment
railway shell

# Check Python version
python --version

# Test API connections
python -c "import os; print('OpenAI Key:', bool(os.getenv('OPENAI_API_KEY')))"

# List files
ls -la

# Check ChromaDB
ls -la chroma_db/
```

## 🔐 Security Best Practices

### Environment Variables
- Never commit API keys to repository
- Use Railway Variables for secrets
- Set ENVIRONMENT=production

### API Keys
- Store securely in Railway dashboard
- Use environment-specific keys
- Monitor API usage

## 🔄 Updates and Maintenance

### Automatic Deployment
Railway automatically redeploys when you:
- Push to main branch
- Update environment variables
- Modify railway.toml

### Database Persistence
- ChromaDB data persists across deployments
- Only recreates when Knowledge Base changes
- Automatic backup and recovery

### Scaling
- Railway handles auto-scaling
- Monitor metrics for performance
- Upgrade plan if needed

## 🌐 Custom Domain (Optional)

1. Go to Railway dashboard
2. Click **Settings** → **Domains**
3. Add your custom domain
4. Update DNS records as shown
5. SSL certificate is auto-generated

## 📊 Performance Optimization

### Model Selection
- **Groq**: Faster, cheaper (recommended for production)
- **OpenAI**: Higher quality, slower
- Switch via UI or config.yaml

### Memory Management
- Railway provides sufficient memory
- ChromaDB is memory-efficient
- Monitor usage in dashboard

### Cost Optimization
- Use Groq for cost-effective responses
- Monitor API usage
- Optimize query frequency

## 🎉 Success Checklist

After deployment, verify:
- [ ] App accessible at Railway URL
- [ ] Context gathering form works
- [ ] Spiritual guidance responses
- [ ] Source attributions showing
- [ ] Model switching works
- [ ] API connection tests pass
- [ ] Database persistence working
- [ ] Health check endpoint responding

## 📧 Support

**Deployment Issues:**
- Check Railway logs and metrics
- Verify environment variables
- Test API connections

**Application Issues:**
- Run local diagnostics
- Check GitHub issues
- Review application logs

**Railway Help:**
- [Railway Docs](https://docs.railway.app)
- Railway Discord community
- Railway support team

---

## 🞆 Your JAI GURU DEV AI is Now Live!

Your spiritual guidance chatbot is ready to serve users worldwide with:
- 🎨 Beautiful saffron-themed interface
- 🧠 Advanced AI-powered responses
- 📚 Rich Sri Sri Ravi Shankar teachings
- ⚡ Lightning-fast performance
- 🌐 Global accessibility

**Share the wisdom with the world! 🙏**

**Jai Guru Dev!**